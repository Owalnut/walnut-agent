package com.walnut.agent.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.walnut.agent.dto.WorkflowDtos;
import com.walnut.agent.entity.WorkflowDefinitionEntity;
import com.walnut.agent.entity.WorkflowExecutionEntity;
import com.walnut.agent.mapper.WorkflowDefinitionMapper;
import com.walnut.agent.mapper.WorkflowExecutionMapper;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.http.HttpStatus;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.Consumer;
import java.util.concurrent.ExecutionException;

@Service
public class WorkflowService {
    private static final Logger log = LoggerFactory.getLogger(WorkflowService.class);
    private final WorkflowDefinitionMapper definitionMapper;
    private final WorkflowExecutionMapper executionMapper;
    private final ObjectMapper objectMapper;
    private final int nodeTimeoutMs;
    private final int retryTimes;
    private final long retryBackoffMs;
    private final int retryBackoffMultiplier;
    private final ChatClientFactory chatClientFactory;

    public WorkflowService(
            WorkflowDefinitionMapper definitionMapper,
            WorkflowExecutionMapper executionMapper,
            ChatClientFactory chatClientFactory,
            ObjectMapper objectMapper,
            @Value("${workflow.node-timeout-ms}") int nodeTimeoutMs,
            @Value("${workflow.retry-times}") int retryTimes,
            @Value("${workflow.retry-backoff-ms}") long retryBackoffMs,
            @Value("${workflow.retry-backoff-multiplier}") int retryBackoffMultiplier
    ) {
        this.definitionMapper = definitionMapper;
        this.executionMapper = executionMapper;
        this.chatClientFactory = chatClientFactory;
        this.objectMapper = objectMapper;
        this.nodeTimeoutMs = nodeTimeoutMs;
        this.retryTimes = retryTimes;
        this.retryBackoffMs = retryBackoffMs;
        this.retryBackoffMultiplier = retryBackoffMultiplier;
    }

    public WorkflowDtos.WorkflowDefinitionResponse saveWorkflow(WorkflowDtos.SaveWorkflowRequest req) {
        WorkflowDefinitionEntity entity = new WorkflowDefinitionEntity();
        entity.setName(req.name());
        entity.setWorkflowJson(writeJson(req.workflow()));
        entity.setIsDraft(req.draft() ? 1 : 0);
        entity.setIsPublished(req.published() ? 1 : 0);
        definitionMapper.insert(entity);
        return new WorkflowDtos.WorkflowDefinitionResponse(entity.getId(), entity.getName(), req.workflow(), req.draft(), req.published());
    }

    public Map<String, Object> getDefaultWorkflow() {
        WorkflowDefinitionEntity entity = definitionMapper.selectOne(
                new LambdaQueryWrapper<WorkflowDefinitionEntity>().eq(WorkflowDefinitionEntity::getIsPublished, 1).last("limit 1")
        );
        if (entity == null) {
            return Map.of(
                    "workflowId", null,
                    "name", "default",
                    "nodes", List.of(
                            Map.of("id", "input-default", "type", "input", "data", Map.of("name", "输入")),
                            Map.of("id", "output-default", "type", "output", "data", Map.of("name", "输出"))
                    ),
                    "edges", List.of(
                            Map.of("id", "e1", "source", "input-default", "target", "output-default")
                    )
            );
        }
        Map<String, Object> graph = readJson(entity.getWorkflowJson());
        graph.put("workflowId", entity.getId());
        graph.put("name", entity.getName());
        return graph;
    }

    public WorkflowDtos.DebugWorkflowResponse debugWorkflow(WorkflowDtos.DebugWorkflowRequest req) {
        return debugWorkflowWithEvents(req, null);
    }

    public WorkflowDtos.DebugWorkflowResponse debugWorkflowWithEvents(
            WorkflowDtos.DebugWorkflowRequest req,
            Consumer<DebugEvent> eventConsumer
    ) {
        WorkflowExecutionEntity execution = new WorkflowExecutionEntity();
        execution.setWorkflowId(req.workflowId() == null ? 0L : req.workflowId());
        execution.setInputText(req.input());
        execution.setStatus("RUNNING");
        executionMapper.insert(execution);
        List<WorkflowDtos.NodeResult> nodeResults = new ArrayList<>();
        emit(eventConsumer, "RUNNING", execution.getId(), Map.of("message", "workflow started"));

        try {
            Map<String, Object> workflowMap = req.workflow() != null ? req.workflow() : getDefaultWorkflow();
            WorkflowDtos.WorkflowGraph graph = objectMapper.convertValue(workflowMap, WorkflowDtos.WorkflowGraph.class);
            WorkflowDtos.DebugOutput output = executeGraph(graph, req.input(), execution.getId(), eventConsumer, nodeResults);
            execution.setStatus("SUCCESS");
            execution.setOutputText(output.text());
            execution.setAudioBase64(output.audioBase64());
            execution.setNodeResults(writeAnyJson(nodeResults));
            execution.setUpdatedAt(LocalDateTime.now());
            executionMapper.updateById(execution);
            emit(eventConsumer, "SUCCESS", execution.getId(), Map.of("output", output));
            return new WorkflowDtos.DebugWorkflowResponse(true, execution.getId(), "SUCCESS", null, output, null);
        } catch (WorkflowNodeException e) {
            execution.setStatus(e.status());
            execution.setErrorMessage(e.getMessage());
            execution.setNodeResults(writeAnyJson(nodeResults));
            execution.setUpdatedAt(LocalDateTime.now());
            executionMapper.updateById(execution);
            return new WorkflowDtos.DebugWorkflowResponse(false, execution.getId(), e.status(), e.code(), null, e.getMessage());
        } catch (Exception e) {
            execution.setStatus("FAILED");
            execution.setErrorMessage(e.getMessage());
            execution.setNodeResults(writeAnyJson(nodeResults));
            execution.setUpdatedAt(LocalDateTime.now());
            executionMapper.updateById(execution);
            emit(eventConsumer, "FAILED", execution.getId(), Map.of("errorCode", "INTERNAL_ERROR", "message", e.getMessage()));
            return new WorkflowDtos.DebugWorkflowResponse(false, execution.getId(), "FAILED", "INTERNAL_ERROR", null, e.getMessage());
        }
    }

    public WorkflowDtos.WorkflowExecutionResponse getExecution(Long executionId) {
        WorkflowExecutionEntity entity = executionMapper.selectById(executionId);
        if (entity == null) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "execution not found");
        }
        List<WorkflowDtos.NodeResult> nodeResults = parseNodeResults(entity.getNodeResults());
        return new WorkflowDtos.WorkflowExecutionResponse(
                entity.getId(),
                entity.getWorkflowId(),
                entity.getInputText(),
                entity.getStatus(),
                entity.getOutputText(),
                entity.getAudioBase64(),
                entity.getErrorMessage(),
                nodeResults
        );
    }

    public List<WorkflowDtos.WorkflowDefinitionResponse> listWorkflows() {
        return definitionMapper.selectList(new LambdaQueryWrapper<WorkflowDefinitionEntity>().orderByDesc(WorkflowDefinitionEntity::getId))
                .stream()
                .map((entity) -> {
                    Map<String, Object> graph = readJson(entity.getWorkflowJson());
                    return new WorkflowDtos.WorkflowDefinitionResponse(
                            entity.getId(),
                            entity.getName(),
                            graph,
                            entity.getIsDraft() != null && entity.getIsDraft() == 1,
                            entity.getIsPublished() != null && entity.getIsPublished() == 1
                    );
                })
                .toList();
    }

    public WorkflowDtos.WorkflowDefinitionResponse getWorkflow(Long workflowId) {
        WorkflowDefinitionEntity entity = definitionMapper.selectById(workflowId);
        if (entity == null) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "workflow not found");
        }
        Map<String, Object> graph = readJson(entity.getWorkflowJson());
        return new WorkflowDtos.WorkflowDefinitionResponse(
                entity.getId(),
                entity.getName(),
                graph,
                entity.getIsDraft() != null && entity.getIsDraft() == 1,
                entity.getIsPublished() != null && entity.getIsPublished() == 1
        );
    }

    public void deleteWorkflow(Long workflowId) {
        int count = definitionMapper.deleteById(workflowId);
        if (count == 0) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "workflow not found");
        }
    }

    public WorkflowDtos.WorkflowExecutionResponse getLatestExecution(Long workflowId) {
        WorkflowExecutionEntity entity = executionMapper.selectOne(
                new LambdaQueryWrapper<WorkflowExecutionEntity>()
                        .eq(WorkflowExecutionEntity::getWorkflowId, workflowId)
                        .orderByDesc(WorkflowExecutionEntity::getId)
                        .last("limit 1")
        );
        if (entity == null) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "execution not found");
        }
        return getExecution(entity.getId());
    }

    private WorkflowDtos.DebugOutput executeGraph(
            WorkflowDtos.WorkflowGraph graph,
            String input,
            Long executionId,
            Consumer<DebugEvent> eventConsumer,
            List<WorkflowDtos.NodeResult> nodeResults
    ) throws Exception {
        List<WorkflowDtos.Node> sorted = topologicalSort(graph.nodes(), graph.edges());
        Map<String, Map<String, String>> values = new HashMap<>();
        Map<String, String> output = Map.of("text", input);
        for (WorkflowDtos.Node node : sorted) {
            emit(eventConsumer, "NODE_START", executionId, Map.of("nodeId", node.id(), "nodeType", node.type()));
            if ("input".equals(node.type())) {
                long startMs = System.currentTimeMillis();
                values.put(node.id(), Map.of("text", input));
                long dur = Math.max(0, System.currentTimeMillis() - startMs);
                nodeResults.add(new WorkflowDtos.NodeResult(node.id(), node.type(), "SUCCESS", dur, input, null));
                emit(eventConsumer, "NODE_SUCCESS", executionId, Map.of("nodeId", node.id(), "text", input));
                continue;
            }
            Map<String, String> prev = findIncomingValue(node.id(), graph.edges(), values);
            if ("llm".equals(node.type())) {
                Map<String, String> promptVars = resolvePromptVariables(node, prev, values, input);
                String prompt = buildPrompt(node, promptVars);
                long startMs = System.currentTimeMillis();
                try {
                    String llmText = executeWithRetry(() -> runWithTimeout(() -> callLlm(node, prompt)));
                    values.put(node.id(), Map.of("text", llmText));
                    long dur = Math.max(0, System.currentTimeMillis() - startMs);
                    nodeResults.add(new WorkflowDtos.NodeResult(node.id(), node.type(), "SUCCESS", dur, llmText, null));
                    emit(eventConsumer, "NODE_SUCCESS", executionId, Map.of("nodeId", node.id(), "text", llmText));
                } catch (WorkflowNodeException e) {
                    long dur = Math.max(0, System.currentTimeMillis() - startMs);
                    nodeResults.add(new WorkflowDtos.NodeResult(node.id(), node.type(), e.status(), dur, null, e.code()));
                    emit(eventConsumer, "NODE_ERROR", executionId, Map.of("nodeId", node.id(), "nodeType", node.type(), "errorCode", e.code(), "message", e.getMessage()));
                    throw e;
                }
            } else if ("tool_tts".equals(node.type())) {
                String text = prev.getOrDefault("text", "");
                long startMs = System.currentTimeMillis();
                try {
                    byte[] wav = executeWithRetry(() -> runWithTimeout(() -> generateSimpleWav()));
                    values.put(node.id(), Map.of("text", text, "audioBase64", Base64.getEncoder().encodeToString(wav), "contentType", "audio/wav"));
                    long dur = Math.max(0, System.currentTimeMillis() - startMs);
                    nodeResults.add(new WorkflowDtos.NodeResult(node.id(), node.type(), "SUCCESS", dur, text, null));
                    emit(eventConsumer, "NODE_SUCCESS", executionId, Map.of("nodeId", node.id(), "audioReady", true));
                } catch (WorkflowNodeException e) {
                    long dur = Math.max(0, System.currentTimeMillis() - startMs);
                    nodeResults.add(new WorkflowDtos.NodeResult(node.id(), node.type(), e.status(), dur, null, e.code()));
                    emit(eventConsumer, "NODE_ERROR", executionId, Map.of("nodeId", node.id(), "nodeType", node.type(), "errorCode", e.code(), "message", e.getMessage()));
                    throw e;
                }
            } else if ("output".equals(node.type())) {
                output = prev;
                values.put(node.id(), output);
                long startMs = System.currentTimeMillis();
                long dur = Math.max(0, System.currentTimeMillis() - startMs);
                nodeResults.add(new WorkflowDtos.NodeResult(node.id(), node.type(), "SUCCESS", dur, output.get("text"), null));
                emit(eventConsumer, "NODE_SUCCESS", executionId, Map.of("nodeId", node.id(), "text", output.get("text")));
            }
        }
        return new WorkflowDtos.DebugOutput(output.get("text"), output.get("audioBase64"), output.getOrDefault("contentType", "audio/wav"));
    }

    private String resolveLlmInput(
            WorkflowDtos.Node node,
            Map<String, String> prev,
            Map<String, Map<String, String>> values,
            String fallbackInput
    ) {
        WorkflowDtos.NodeData data = node.data();
        String sourceNodeId = data == null ? null : data.inputSourceNodeId();
        if (sourceNodeId != null && !sourceNodeId.isBlank()) {
            Map<String, String> sourceVal = values.get(sourceNodeId);
            if (sourceVal != null) {
                String sourceText = sourceVal.get("text");
                if (sourceText != null && !sourceText.isBlank()) {
                    return sourceText;
                }
            }
        }
        String prevText = prev.get("text");
        if (prevText != null && !prevText.isBlank()) {
            return prevText;
        }
        return fallbackInput == null ? "" : fallbackInput;
    }

    private Map<String, String> resolvePromptVariables(
            WorkflowDtos.Node node,
            Map<String, String> prev,
            Map<String, Map<String, String>> values,
            String fallbackInput
    ) {
        Map<String, String> vars = new LinkedHashMap<>();
        String defaultInput = resolveLlmInput(node, prev, values, fallbackInput);
        vars.put("input", defaultInput == null ? "" : defaultInput);
        WorkflowDtos.NodeData data = node.data();
        if (data == null || data.inputParams() == null) {
            return vars;
        }
        for (WorkflowDtos.NodeInputParam param : data.inputParams()) {
            if (param == null || param.name() == null || param.name().isBlank()) {
                continue;
            }
            String value = resolveInputParamValue(param, values);
            vars.put(param.name(), value == null ? "" : value);
        }
        return vars;
    }

    private String resolveInputParamValue(WorkflowDtos.NodeInputParam param, Map<String, Map<String, String>> values) {
        if ("reference".equals(param.type())) {
            String ref = param.value();
            if (ref == null || ref.isBlank()) {
                return "";
            }
            String[] parts = ref.split("\\.", 2);
            String nodeId = parts[0];
            String field = parts.length > 1 ? parts[1] : "text";
            return values.getOrDefault(nodeId, Map.of()).getOrDefault(field, "");
        }
        return param.value() == null ? "" : param.value();
    }

    private String buildPrompt(WorkflowDtos.Node node, Map<String, String> promptVars) {
        WorkflowDtos.NodeData data = node.data();
        String template = data == null ? null : data.promptTemplate();
        if (template == null || template.isBlank()) {
            return promptVars.getOrDefault("input", "");
        }
        String prompt = template;
        for (Map.Entry<String, String> entry : promptVars.entrySet()) {
            prompt = prompt.replace("{{" + entry.getKey() + "}}", entry.getValue() == null ? "" : entry.getValue());
        }
        return prompt;
    }

    private static String defaultLlmModel(String provider) {
        if (provider == null || provider.isBlank()) {
            return "deepseek-chat";
        }
        String p = provider.trim().toLowerCase(Locale.ROOT);
        if ("dashscope".equals(p) || "qwen".equals(p)) {
            return "qwen-plus";
        }
        return "deepseek-chat";
    }

    private String callLlm(WorkflowDtos.Node node, String prompt) throws Exception {
        WorkflowDtos.NodeData data = node.data();
        String baseUrl = data == null ? null : data.baseUrl();
        String apiKey = data == null ? null : data.apiKey();
        String provider = data == null ? null : data.provider();
        String model =
                data == null || data.model() == null || data.model().isBlank()
                        ? defaultLlmModel(provider)
                        : data.model();
        Double temperature = data == null || data.temperature() == null ? 0.7 : data.temperature();

        if (baseUrl == null || baseUrl.isBlank()) {
            throw new WorkflowNodeException("NODE_CONFIG_ERROR", "FAILED", "LLM baseUrl is required", null);
        }
        if (apiKey == null || apiKey.isBlank()) {
            throw new WorkflowNodeException("NODE_CONFIG_ERROR", "FAILED", "LLM apiKey is required", null);
        }
        try {
            log.info(
                    "LLM node execute: baseUrl={}, model={}, temperature={}, promptLen={}",
                    baseUrl,
                    model,
                    temperature,
                    prompt == null ? 0 : prompt.length()
            );
            ChatClient chatClient = chatClientFactory.createDeepSeekClient(baseUrl, apiKey, model, temperature);
            String content = chatClient.prompt()
                    .user(prompt == null ? "" : prompt)
                    .call()
                    .content();
            log.info("LLM node response: contentLen={}", content == null ? 0 : content.length());
            if (content == null || content.isBlank()) {
                throw new WorkflowNodeException("LLM_EMPTY_RESPONSE", "FAILED", "LLM returned empty content", null);
            }
            return content;
        } catch (WorkflowNodeException e) {
            log.warn("LLM node failed (known error): code={}, status={}, message={}", e.code(), e.status(), e.getMessage());
            throw e;
        } catch (Exception e) {
            log.error("LLM provider error: root={}", getRootMessage(e));
            throw new WorkflowNodeException(
                    "LLM_PROVIDER_ERROR",
                    "FAILED",
                    "Spring AI LLM call failed: " + getRootMessage(e),
                    e
            );
        }
    }

    private String getRootMessage(Throwable t) {
        Throwable cur = t;
        while (cur.getCause() != null) {
            cur = cur.getCause();
        }
        String msg = cur.getMessage();
        return msg == null || msg.isBlank() ? cur.getClass().getSimpleName() : msg;
    }

    private void emit(Consumer<DebugEvent> consumer, String type, Long executionId, Object payload) {
        if (consumer == null) return;
        consumer.accept(new DebugEvent(type, executionId, payload));
    }

    private <T> T executeWithRetry(Callable<T> callable) throws Exception {
        Exception last = null;
        long backoff = retryBackoffMs;
        for (int i = 0; i < retryTimes; i++) {
            try {
                return callable.call();
            } catch (Exception e) {
                // Provider/config errors are deterministic; fail fast and keep original message.
                if (e instanceof WorkflowNodeException wne &&
                        ("NODE_CONFIG_ERROR".equals(wne.code())
                                || "LLM_PROVIDER_ERROR".equals(wne.code())
                                || "LLM_EMPTY_RESPONSE".equals(wne.code())
                                || "LLM_RESPONSE_PARSE_ERROR".equals(wne.code()))) {
                    throw wne;
                }
                last = e;
                if (i < retryTimes - 1 && backoff > 0) {
                    Thread.sleep(backoff);
                    backoff = backoff * Math.max(1, retryBackoffMultiplier);
                }
            }
        }
        if (last instanceof WorkflowNodeException wne) {
            throw wne;
        }
        throw new WorkflowNodeException("NODE_RETRY_EXHAUSTED", "FAILED", "Node execution failed after retries", last);
    }

    private <T> T runWithTimeout(Callable<T> callable) throws Exception {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Future<T> future = executor.submit(callable);
        try {
            return future.get(nodeTimeoutMs, TimeUnit.MILLISECONDS);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause();
            if (cause instanceof WorkflowNodeException wne) {
                throw wne;
            }
            if (cause instanceof Exception ex) {
                throw ex;
            }
            throw new WorkflowNodeException("INTERNAL_ERROR", "FAILED", "Unexpected node execution error", e);
        } catch (TimeoutException e) {
            future.cancel(true);
            throw new WorkflowNodeException("NODE_TIMEOUT", "TIMEOUT", "Node execution timeout", e);
        } catch (InterruptedException e) {
            future.cancel(true);
            Thread.currentThread().interrupt();
            throw new WorkflowNodeException("NODE_INTERRUPTED", "INTERRUPTED", "Node execution interrupted", e);
        } finally {
            executor.shutdownNow();
        }
    }

    private List<WorkflowDtos.Node> topologicalSort(List<WorkflowDtos.Node> nodes, List<WorkflowDtos.Edge> edges) {
        Map<String, WorkflowDtos.Node> nodeMap = new HashMap<>();
        Map<String, Integer> inDegree = new HashMap<>();
        Map<String, List<String>> graph = new HashMap<>();
        for (WorkflowDtos.Node node : nodes) {
            nodeMap.put(node.id(), node);
            inDegree.put(node.id(), 0);
            graph.put(node.id(), new ArrayList<>());
        }
        for (WorkflowDtos.Edge edge : edges) {
            graph.get(edge.source()).add(edge.target());
            inDegree.put(edge.target(), inDegree.get(edge.target()) + 1);
        }
        Deque<String> queue = new ArrayDeque<>();
        inDegree.forEach((id, degree) -> {
            if (degree == 0) queue.offer(id);
        });
        List<WorkflowDtos.Node> result = new ArrayList<>();
        while (!queue.isEmpty()) {
            String id = queue.poll();
            result.add(nodeMap.get(id));
            for (String next : graph.get(id)) {
                inDegree.put(next, inDegree.get(next) - 1);
                if (inDegree.get(next) == 0) queue.offer(next);
            }
        }
        if (result.size() != nodes.size()) throw new RuntimeException("Workflow has cycle");
        return result;
    }

    private Map<String, String> findIncomingValue(String nodeId, List<WorkflowDtos.Edge> edges, Map<String, Map<String, String>> values) {
        for (WorkflowDtos.Edge edge : edges) {
            if (edge.target().equals(nodeId)) {
                return values.getOrDefault(edge.source(), Map.of());
            }
        }
        return Map.of();
    }

    private Map<String, Object> readJson(String json) {
        try {
            return objectMapper.readValue(json, new TypeReference<>() {});
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Invalid workflow json");
        }
    }

    private String writeJson(Map<String, Object> map) {
        try {
            return objectMapper.writeValueAsString(map);
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Cannot serialize workflow json");
        }
    }

    private String writeAnyJson(Object value) {
        try {
            return objectMapper.writeValueAsString(value);
        } catch (JsonProcessingException e) {
            return null;
        }
    }

    private List<WorkflowDtos.NodeResult> parseNodeResults(String json) {
        if (json == null || json.isBlank()) {
            return List.of();
        }
        try {
            return objectMapper.readValue(json, new TypeReference<List<WorkflowDtos.NodeResult>>() {});
        } catch (Exception e) {
            return List.of();
        }
    }

    private byte[] generateSimpleWav() {
        int sampleRate = 16000;
        int durationMs = 800;
        int totalSamples = sampleRate * durationMs / 1000;
        byte[] pcm = new byte[totalSamples * 2];
        double frequency = 440.0;
        for (int i = 0; i < totalSamples; i++) {
            short value = (short) (Math.sin(2 * Math.PI * frequency * i / sampleRate) * 32767 * 0.2);
            pcm[i * 2] = (byte) (value & 0xff);
            pcm[i * 2 + 1] = (byte) ((value >> 8) & 0xff);
        }
        return wavWrap(pcm, sampleRate, 1, 16);
    }

    private byte[] wavWrap(byte[] pcm, int sampleRate, int channels, int bitsPerSample) {
        int byteRate = sampleRate * channels * bitsPerSample / 8;
        int blockAlign = channels * bitsPerSample / 8;
        ByteBuffer buffer = ByteBuffer.allocate(44 + pcm.length).order(ByteOrder.LITTLE_ENDIAN);
        buffer.put("RIFF".getBytes());
        buffer.putInt(36 + pcm.length);
        buffer.put("WAVE".getBytes());
        buffer.put("fmt ".getBytes());
        buffer.putInt(16);
        buffer.putShort((short) 1);
        buffer.putShort((short) channels);
        buffer.putInt(sampleRate);
        buffer.putInt(byteRate);
        buffer.putShort((short) blockAlign);
        buffer.putShort((short) bitsPerSample);
        buffer.put("data".getBytes());
        buffer.putInt(pcm.length);
        buffer.put(pcm);
        return buffer.array();
    }

    private static class WorkflowNodeException extends RuntimeException {
        private final String code;
        private final String status;

        private WorkflowNodeException(String code, String status, String message, Throwable cause) {
            super(message, cause);
            this.code = code;
            this.status = status;
        }

        public String code() {
            return code;
        }

        public String status() {
            return status;
        }
    }

    public record DebugEvent(String type, Long executionId, Object payload) {}
}
