import { DragEvent, useEffect, useMemo, useRef, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  applyNodeChanges,
  Edge,
  Handle,
  MarkerType,
  Node,
  NodeProps,
  Position,
  ReactFlowInstance
} from "reactflow";
import "reactflow/dist/style.css";

const API_BASE = "http://localhost:8787";
const WS_BASE = "ws://localhost:8787/ws/debug";

type NodeType = "input" | "llm" | "tool_tts" | "output";
type WorkflowNodeMeta = {
  id: string;
  type: NodeType;
  data?: {
    name?: string;
    model?: string;
    voice?: string;
    provider?: string;
    baseUrl?: string;
    apiKey?: string;
    temperature?: number;
    promptTemplate?: string;
    inputSourceNodeId?: string;
    inputParams?: DeepSeekInputParam[];
    outputParams?: DeepSeekOutputParam[];
  };
};
type WorkflowEdge = { id: string; source: string; target: string };
type Workflow = { nodes: WorkflowNodeMeta[]; edges: WorkflowEdge[] };
type DebugOutput = { text?: string; audioBase64?: string; contentType?: string };
type WsEvent = { type: string; executionId?: number; payload?: Record<string, unknown> };
type OutputParamType = "input" | "reference";
type OutputParam = { id: string; name: string; type: OutputParamType; value: string };
type DeepSeekInputParamType = "input" | "reference";
type DeepSeekInputParam = { id: string; name: string; type: DeepSeekInputParamType; value: string };
type DeepSeekOutputParam = { id: string; name: string; valueType: "string"; description: string };

type NodeResult = {
  nodeId: string;
  nodeType: NodeType;
  status: string;
  durationMs: number | null;
  text: string | null;
  errorCode: string | null;
};
type WorkflowExecutionResponse = {
  id: number;
  workflowId: number;
  inputText: string;
  status: string;
  outputText: string | null;
  audioBase64: string | null;
  errorMessage: string | null;
  nodeResults: NodeResult[];
};
type WorkflowDefinitionResponse = {
  id: number;
  name: string;
  workflow: Workflow;
  draft: boolean;
  published: boolean;
};
type DeepSeekConfig = {
  baseUrl: string;
  apiKey: string;
  temperature: number;
  model: string;
  promptTemplate: string;
  inputParams: DeepSeekInputParam[];
  outputParams: DeepSeekOutputParam[];
};

/** 通义千问（DashScope OpenAI 兼容）配置，字段与 DeepSeek 同源，便于工作流序列化 */
type QwenConfig = DeepSeekConfig;

const DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1";
const QWEN_MODEL_PRESETS = ["qwen-turbo", "qwen-plus", "qwen-max", "qwen-long", "qwen-flash"] as const;

type FlowNode = Node<{ label: string; nodeType: NodeType }, "workflowNode"> & { meta: WorkflowNodeMeta };

function WorkflowNodeView({ data, isConnectable }: NodeProps<{ label: string; nodeType: NodeType }>) {
  const supportsSource = data.nodeType === "input" || data.nodeType === "llm" || data.nodeType === "tool_tts";
  const supportsTarget = data.nodeType === "llm" || data.nodeType === "tool_tts" || data.nodeType === "output";

  return (
    <div className="workflow-node" style={{ pointerEvents: "all" }}>
      <div className="workflow-node-label">{data.label}</div>
      {supportsTarget && <Handle type="target" position={Position.Left} id="in" isConnectable={isConnectable} />}
      {supportsSource && <Handle type="source" position={Position.Right} id="out" isConnectable={isConnectable} />}
    </div>
  );
}

const NODE_TYPES = { workflowNode: WorkflowNodeView } as const;

const llmNodes: Array<{ type: NodeType; label: string; icon: string }> = [
  { type: "llm", label: "DeepSeek", icon: "🧠" },
  { type: "llm", label: "通义千问", icon: "✨" },
  { type: "llm", label: "AI Ping", icon: "🚀" },
  { type: "llm", label: "智谱", icon: "🧝" }
];
const baseNodes: Array<{ type: NodeType; label: string; icon: string }> = [
  { type: "input", label: "输入", icon: "⌨️" },
  { type: "output", label: "输出", icon: "🔚" }
];
const toolNodes: Array<{ type: NodeType; label: string; icon: string }> = [
  { type: "tool_tts", label: "超拟人音频合成", icon: "🎙️" }
];

const nodeTypeLabel: Record<NodeType, string> = {
  input: "输入",
  llm: "大模型",
  tool_tts: "工具",
  output: "输出"
};

export default function App() {
  const DEBUG_DRAWER_MIN_HEIGHT = 64;
  const DEBUG_DRAWER_MIN_OPEN_HEIGHT = 220;
  const DEBUG_DRAWER_MAX_HEIGHT_RATIO = 0.85;
  const [token, setToken] = useState("");
  const [nodes, setNodes] = useState<FlowNode[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [selectedNode, setSelectedNode] = useState<WorkflowNodeMeta | null>(null);
  const [debugOpen, setDebugOpen] = useState(true);
  const [debugDrawerHeight, setDebugDrawerHeight] = useState(320);
  const [isResizingDebugDrawer, setIsResizingDebugDrawer] = useState(false);
  const [debugInput, setDebugInput] = useState("你好，帮我生成一期关于 AI Agent 的播客开场白。");
  const [debugResult, setDebugResult] = useState<DebugOutput | null>(null);
  const [loading, setLoading] = useState(false);
  const [currentWorkflowId, setCurrentWorkflowId] = useState<number | null>(null);
  const [latestExecution, setLatestExecution] = useState<WorkflowExecutionResponse | null>(null);
  const [liveNodeResults, setLiveNodeResults] = useState<NodeResult[]>([]);
  const [executionStatus, setExecutionStatus] = useState<string>("");
  const [executionError, setExecutionError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [outputConfigs, setOutputConfigs] = useState<Record<string, OutputParam[]>>({});
  const [outputTemplates, setOutputTemplates] = useState<Record<string, string>>({});
  const [deepSeekConfig, setDeepSeekConfig] = useState<DeepSeekConfig>({
    baseUrl: "",
    apiKey: "",
    temperature: 0.7,
    model: "deepseek-chat",
    promptTemplate: "",
    inputParams: [],
    outputParams: []
  });
  const [qwenConfig, setQwenConfig] = useState<QwenConfig>({
    baseUrl: DEFAULT_QWEN_BASE_URL,
    apiKey: "",
    temperature: 0.7,
    model: "qwen-plus",
    promptTemplate: "",
    inputParams: [],
    outputParams: []
  });
  const [loadModalOpen, setLoadModalOpen] = useState(false);
  const [workflowsList, setWorkflowsList] = useState<WorkflowDefinitionResponse[]>([]);
  const [workflowsLoading, setWorkflowsLoading] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const flowRef = useRef<ReactFlowInstance | null>(null);
  const debugResizeStartRef = useRef<{ startY: number; startHeight: number } | null>(null);

  useEffect(() => {
    void loginAndLoad();
    return () => wsRef.current?.close();
  }, []);

  useEffect(() => {
    if (!isResizingDebugDrawer) return;

    const onMouseMove = (event: MouseEvent) => {
      const start = debugResizeStartRef.current;
      if (!start) return;
      const delta = start.startY - event.clientY;
      const next = start.startHeight + delta;
      const maxHeight = Math.floor(window.innerHeight * DEBUG_DRAWER_MAX_HEIGHT_RATIO);
      const minHeight = debugOpen ? DEBUG_DRAWER_MIN_OPEN_HEIGHT : DEBUG_DRAWER_MIN_HEIGHT;
      setDebugDrawerHeight(Math.max(minHeight, Math.min(next, maxHeight)));
    };

    const onMouseUp = () => {
      setIsResizingDebugDrawer(false);
      debugResizeStartRef.current = null;
    };

    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }, [isResizingDebugDrawer, debugOpen]);

  async function loginAndLoad(): Promise<void> {
    const loginResp = await fetch(`${API_BASE}/api/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: "admin", password: "admin123" })
    });
    const loginData = (await loginResp.json()) as { token: string };
    const nextToken = loginData.token;
    setToken(nextToken);

    const savedWorkflowIdRaw = localStorage.getItem("walnutAgent.currentWorkflowId");
    const savedWorkflowId = savedWorkflowIdRaw ? Number(savedWorkflowIdRaw) : null;

    if (savedWorkflowId) {
      try {
        await handleLoadWorkflow(savedWorkflowId, nextToken);
        return;
      } catch {
        // fallback to default
      }
    }

    const workflowResp = await fetch(`${API_BASE}/api/workflows/default`, {
      headers: { Authorization: `Bearer ${nextToken}` }
    });
    const raw = (await workflowResp.json()) as Workflow & { workflowId?: number | null; name?: string };
    const workflow = raw;
    const flowNodes = workflow.nodes.map((node, index) => ({
      id: node.id,
      type: "workflowNode" as const,
      position: { x: 280, y: 60 + index * 125 },
      data: { label: node.data?.name || nodeTypeLabel[node.type] || node.type, nodeType: node.type },
      draggable: true,
      meta: node
    }));
    const flowEdges = workflow.edges.map((edge) => ({
      ...edge,
      animated: true,
      markerEnd: { type: MarkerType.ArrowClosed, color: "#6b7280" }
    }));
    setNodes(flowNodes);
    setEdges(flowEdges);
    setCurrentWorkflowId(raw.workflowId ?? null);
    if (raw.workflowId) {
      localStorage.setItem("walnutAgent.currentWorkflowId", String(raw.workflowId));
    }
    if (raw.workflowId) {
      await loadLatestExecution(raw.workflowId, nextToken);
    } else {
      setLatestExecution(null);
      setLiveNodeResults([]);
      setExecutionStatus("");
      setExecutionError(null);
      setDebugResult(null);
    }
  }

  function handleNewWorkflow() {
    setNodes([]);
    setEdges([]);
    setSelectedNode(null);
    setOutputConfigs({});
    setOutputTemplates({});
    setCurrentWorkflowId(null);
    localStorage.removeItem("walnutAgent.currentWorkflowId");
    setLatestExecution(null);
    setLiveNodeResults([]);
    setExecutionStatus("");
    setExecutionError(null);
    setDebugResult(null);
  }

  async function loadLatestExecution(workflowId: number, authToken: string = token): Promise<void> {
    try {
      const resp = await fetch(`${API_BASE}/api/workflows/${workflowId}/latest-execution`, {
        headers: { Authorization: `Bearer ${authToken}` }
      });
      if (!resp.ok) {
        setLatestExecution(null);
        setLiveNodeResults([]);
        setExecutionStatus("");
        setExecutionError(null);
        return;
      }
      const latest = (await resp.json()) as WorkflowExecutionResponse;
      setLatestExecution(latest);
      setExecutionStatus(latest.status);
      setExecutionError(latest.errorMessage);
      setLiveNodeResults(latest.nodeResults ?? []);
      if (latest.outputText || latest.audioBase64) {
        setDebugResult({
          text: latest.outputText ?? "",
          audioBase64: latest.audioBase64 ?? undefined,
          contentType: "audio/wav"
        });
      } else {
        setDebugResult(null);
      }
    } catch (e) {
      setLatestExecution(null);
      setLiveNodeResults([]);
      setExecutionStatus("");
      setExecutionError(null);
    }
  }

  async function loadWorkflowsList(authToken: string = token): Promise<void> {
    if (!authToken) return;
    setWorkflowsLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/api/workflows`, {
        headers: { Authorization: `Bearer ${authToken}` }
      });
      if (!resp.ok) throw new Error(`list workflows failed: ${resp.status}`);
      const list = (await resp.json()) as WorkflowDefinitionResponse[];
      setWorkflowsList(list);
    } catch (e) {
      setWorkflowsList([]);
    } finally {
      setWorkflowsLoading(false);
    }
  }

  async function handleLoadWorkflow(workflowId: number, authToken: string = token): Promise<void> {
    try {
      const resp = await fetch(`${API_BASE}/api/workflows/${workflowId}`, {
        headers: { Authorization: `Bearer ${authToken}` }
      });
      if (!resp.ok) throw new Error(`load workflow failed: ${resp.status}`);
      const def = (await resp.json()) as WorkflowDefinitionResponse;
      const workflow = def.workflow;
      const flowNodes = workflow.nodes.map((node, index) => ({
        id: node.id,
        type: "workflowNode" as const,
        position: { x: 280, y: 60 + index * 125 },
        data: { label: node.data?.name || nodeTypeLabel[node.type] || node.type, nodeType: node.type },
        draggable: true,
        meta: node
      }));
      const flowEdges = workflow.edges.map((edge) => ({
        ...edge,
        animated: true,
        markerEnd: { type: MarkerType.ArrowClosed, color: "#6b7280" }
      }));
      setNodes(flowNodes);
      setEdges(flowEdges);
      setSelectedNode(null);
      setOutputConfigs({});
      setOutputTemplates({});
      setCurrentWorkflowId(workflowId);
      localStorage.setItem("walnutAgent.currentWorkflowId", String(workflowId));
      setLoadModalOpen(false);
      await loadLatestExecution(workflowId, authToken);
    } catch (e) {
      alert(e instanceof Error ? e.message : String(e));
    }
  }

  async function handleDeleteWorkflow(workflowId: number, authToken: string = token): Promise<void> {
    if (!authToken) {
      alert("请先登录/加载工作流");
      return;
    }
    const ok = confirm(`确定删除该工作流（ID=${workflowId}）吗？`);
    if (!ok) return;
    try {
      const resp = await fetch(`${API_BASE}/api/workflows/${workflowId}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${authToken}` }
      });
      if (!resp.ok) {
        const text = await resp.text().catch(() => "");
        throw new Error(text || `delete workflow failed: ${resp.status}`);
      }

      // 如果删除的是当前工作流，重置画布与本地状态
      if (currentWorkflowId === workflowId) {
        handleNewWorkflow();
        setLoadModalOpen(false);
      }
      await loadWorkflowsList(authToken);
    } catch (e) {
      alert(e instanceof Error ? e.message : String(e));
    }
  }

  const workflowPayload = useMemo(
    () => ({
      nodes: nodes.map((node) => {
        const meta = (node as unknown as { meta?: WorkflowNodeMeta }).meta;
        if (meta) return meta;
        // 兜底：如果自定义字段 meta 在某些更新流程中丢失，用 data 反推
        return {
          id: node.id,
          type: (node.data as unknown as { nodeType?: NodeType }).nodeType ?? "input",
          data: {
            name: (node.data as unknown as { label?: string }).label ?? node.id
          }
        } satisfies WorkflowNodeMeta;
      }),
      edges: edges.map(({ id, source, target }) => ({ id, source, target }))
    }),
    [nodes, edges]
  );

  const selectedOutputParams = selectedNode && selectedNode.type === "output" ? outputConfigs[selectedNode.id] || [] : [];
  const selectedOutputTemplate =
    selectedNode && selectedNode.type === "output" ? outputTemplates[selectedNode.id] ?? "{{output}}" : "{{output}}";
  const isQwenNode =
    selectedNode?.type === "llm" &&
    (selectedNode.data?.provider === "dashscope" ||
      selectedNode.data?.provider === "qwen" ||
      /通义|千问|qwen/i.test(selectedNode.data?.name ?? ""));

  const isDeepSeekNode =
    selectedNode?.type === "llm" &&
    !isQwenNode &&
    ((selectedNode.data?.name ?? "").toLowerCase().includes("deepseek") ||
      selectedNode.data?.provider === "deepseek" ||
      selectedNode.data?.model === "deepseek-chat");

  const referenceOptions = useMemo(() => {
    return nodes
      .filter((n) => {
        const metaType = (n as unknown as { meta?: WorkflowNodeMeta }).meta?.type;
        const dataType = (n.data as unknown as { nodeType?: NodeType }).nodeType;
        return (metaType ?? dataType) !== "output";
      })
      .flatMap((n) => [
        { label: `${n.data.label}.text`, value: `${n.id}.text` },
        { label: `${n.data.label}.audioBase64`, value: `${n.id}.audioBase64` }
      ]);
  }, [nodes]);

  const deepSeekReferenceOptions = useMemo(() => {
    if (!selectedNode) return [];
    const previousNodeIds = new Set(
      edges.filter((e) => e.target === selectedNode.id).map((e) => e.source)
    );
    return nodes
      .filter((n) => previousNodeIds.has(n.id))
      .flatMap((n) => [
        { label: `${n.data.label}.text`, value: `${n.id}.text` },
        { label: `${n.data.label}.audioBase64`, value: `${n.id}.audioBase64` }
      ]);
  }, [edges, nodes, selectedNode]);

  useEffect(() => {
    if (!isDeepSeekNode || !selectedNode) return;
    const data = selectedNode.data ?? {};
    setDeepSeekConfig({
      baseUrl: data.baseUrl ?? "",
      apiKey: data.apiKey ?? "",
      temperature: typeof data.temperature === "number" ? data.temperature : 0.7,
      model: "deepseek-chat",
      promptTemplate: data.promptTemplate ?? "",
      inputParams: data.inputParams ?? [],
      outputParams: data.outputParams ?? []
    });
  }, [isDeepSeekNode, selectedNode]);

  useEffect(() => {
    if (!isQwenNode || !selectedNode) return;
    const data = selectedNode.data ?? {};
    setQwenConfig({
      baseUrl: data.baseUrl?.trim() ? data.baseUrl! : DEFAULT_QWEN_BASE_URL,
      apiKey: data.apiKey ?? "",
      temperature: typeof data.temperature === "number" ? data.temperature : 0.7,
      model: data.model?.trim() ? data.model! : "qwen-plus",
      promptTemplate: data.promptTemplate ?? "",
      inputParams: data.inputParams ?? [],
      outputParams: data.outputParams ?? []
    });
  }, [isQwenNode, selectedNode]);

  function persistSelectedNodeData(nodeId: string, nextData: WorkflowNodeMeta["data"]) {
    setNodes((prev) =>
      prev.map((node) => {
        if (node.id !== nodeId) return node;
        const prevMeta = (node as unknown as { meta?: WorkflowNodeMeta }).meta;
        const mergedMeta: WorkflowNodeMeta = {
          id: nodeId,
          type: prevMeta?.type ?? ((node.data as unknown as { nodeType?: NodeType }).nodeType ?? "input"),
          data: {
            ...(prevMeta?.data ?? {}),
            ...(nextData ?? {})
          }
        };
        return {
          ...node,
          meta: mergedMeta,
          data: {
            ...node.data,
            label: mergedMeta.data?.name ?? node.data.label
          }
        };
      })
    );
    setSelectedNode((prev) => {
      if (!prev || prev.id !== nodeId) return prev;
      return {
        ...prev,
        data: {
          ...(prev.data ?? {}),
          ...(nextData ?? {})
        }
      };
    });
  }

  function saveDeepSeekConfig() {
    if (!selectedNode) return;
    const baseUrl = deepSeekConfig.baseUrl.trim();
    if (!baseUrl) {
      alert("DeepSeek 模型接口地址为必填项");
      return;
    }
    const payload: WorkflowNodeMeta["data"] = {
      ...(selectedNode.data ?? {}),
      name: selectedNode.data?.name ?? "DeepSeek",
      provider: "deepseek",
      baseUrl,
      apiKey: deepSeekConfig.apiKey,
      temperature: Number(deepSeekConfig.temperature),
      model: "deepseek-chat",
      promptTemplate: deepSeekConfig.promptTemplate,
      inputParams: deepSeekConfig.inputParams,
      outputParams: deepSeekConfig.outputParams
    };
    persistSelectedNodeData(selectedNode.id, payload);
    setLogs((prev) => [...prev, `[CONFIG] DeepSeek config saved for ${selectedNode.id}`]);
  }

  function saveQwenConfig() {
    if (!selectedNode) return;
    const baseUrl = qwenConfig.baseUrl.trim();
    if (!baseUrl) {
      alert("请填写 DashScope 兼容模式 API 根地址");
      return;
    }
    const model = qwenConfig.model.trim();
    if (!model) {
      alert("请选择或填写模型名称（如 qwen-plus）");
      return;
    }
    const payload: WorkflowNodeMeta["data"] = {
      ...(selectedNode.data ?? {}),
      name: selectedNode.data?.name ?? "通义千问",
      provider: "dashscope",
      baseUrl,
      apiKey: qwenConfig.apiKey,
      temperature: Number(qwenConfig.temperature),
      model,
      promptTemplate: qwenConfig.promptTemplate,
      inputParams: qwenConfig.inputParams,
      outputParams: qwenConfig.outputParams
    };
    persistSelectedNodeData(selectedNode.id, payload);
    setLogs((prev) => [...prev, `[CONFIG] 通义千问 config saved for ${selectedNode.id}`]);
  }

  async function handleDebug(): Promise<void> {
    if (!token) return;
    setLoading(true);
    setLogs([]);
    setDebugResult(null);
    setLatestExecution(null);
    setLiveNodeResults([]);
    setExecutionStatus("RUNNING");
    setExecutionError(null);
    wsRef.current?.close();

    const ws = new WebSocket(`${WS_BASE}?token=${encodeURIComponent(token)}`);
    wsRef.current = ws;
    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          type: "START_DEBUG",
          payload: { input: debugInput, workflow: workflowPayload, workflowId: currentWorkflowId }
        })
      );
    };
    ws.onmessage = async (event) => {
      const message = JSON.parse(event.data) as WsEvent;
      const payload = message.payload ?? {};
      setLogs((prev) => [...prev, `[${message.type}] ${JSON.stringify(payload)}`]);

      if (message.type === "RUNNING") {
        setExecutionStatus("RUNNING");
        setExecutionError(null);
        setLiveNodeResults([]);
      } else if (message.type === "NODE_START") {
        const nodeId = (payload as any).nodeId as string;
        const nodeType = (payload as any).nodeType as NodeType;
        setLiveNodeResults((prev) => {
          if (prev.find((p) => p.nodeId === nodeId)) return prev;
          return [...prev, { nodeId, nodeType, status: "RUNNING", durationMs: null, text: null, errorCode: null }];
        });
      } else if (message.type === "NODE_SUCCESS") {
        const nodeId = (payload as any).nodeId as string;
        const text = (payload as any).text as string | undefined;
        setLiveNodeResults((prev) =>
          prev.map((n) => (n.nodeId === nodeId ? { ...n, status: "SUCCESS", text: text ?? n.text } : n))
        );
      } else if (message.type === "NODE_ERROR") {
        const nodeId = (payload as any).nodeId as string;
        const nodeType = (payload as any).nodeType as NodeType;
        const errorCode = (payload as any).errorCode as string | null;
        setLiveNodeResults((prev) =>
          prev.map((n) =>
            n.nodeId === nodeId ? { ...n, nodeType: nodeType ?? n.nodeType, status: "FAILED", errorCode: errorCode ?? n.errorCode } : n
          )
        );
      }
      if (message.type === "COMPLETED") {
        const output = (payload as any).output as DebugOutput | undefined;
        setDebugResult(output ?? null);
        if (output?.audioBase64) {
          const audio = new Audio(`data:${output.contentType};base64,${output.audioBase64}`);
          await audio.play();
        }
        if (message.executionId) {
          await loadExecutionById(message.executionId, token);
        }
        setExecutionStatus((payload as any).executionStatus ?? "SUCCESS");
        setLoading(false);
        ws.close();
      } else if (message.type === "FAILED") {
        if (message.executionId) {
          await loadExecutionById(message.executionId, token);
        }
        setExecutionStatus((payload as any).executionStatus ?? "FAILED");
        setExecutionError((payload as any).error ?? (payload as any).errorCode ?? "执行失败");
        setLoading(false);
        ws.close();
      }
    };
    ws.onerror = () => {
      setLogs((prev) => [...prev, "[ERROR] websocket connection failed"]);
      setLoading(false);
    };
    ws.onclose = () => {
      setLoading(false);
    };
  }

  async function loadExecutionById(executionId: number, authToken: string): Promise<void> {
    const resp = await fetch(`${API_BASE}/api/workflows/executions/${executionId}`, {
      headers: { Authorization: `Bearer ${authToken}` }
    });
    if (!resp.ok) return;
    const exec = (await resp.json()) as WorkflowExecutionResponse;
    setLatestExecution(exec);
    setExecutionStatus(exec.status);
    setExecutionError(exec.errorMessage);
    setLiveNodeResults(exec.nodeResults ?? []);
    if (exec.outputText || exec.audioBase64) {
      setDebugResult({
        text: exec.outputText ?? "",
        audioBase64: exec.audioBase64 ?? undefined,
        contentType: "audio/wav"
      });
    } else {
      setDebugResult(null);
    }
  }

  async function handleSaveWorkflow(): Promise<void> {
    if (!token) {
      alert("请先登录/加载工作流");
      return;
    }
    const name = window.prompt("请输入工作流名称", `workflow-${Date.now()}`);
    if (!name) return;
    setSaving(true);
    try {
      const resp = await fetch(`${API_BASE}/api/workflows`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({
          name,
          workflow: workflowPayload,
          draft: true,
          published: false
        })
      });
      if (!resp.ok) {
        const text = await resp.text().catch(() => "");
        throw new Error(text || `save failed: ${resp.status}`);
      }
      const saved = (await resp.json()) as { id?: number };
      const savedId = saved.id ?? null;
      setLogs((prev) => [...prev, `[SAVE] ok id=${savedId ?? "-"}`]);
      if (savedId) {
        setCurrentWorkflowId(savedId);
        localStorage.setItem("walnutAgent.currentWorkflowId", String(savedId));
        setLatestExecution(null);
        setLiveNodeResults([]);
        setExecutionStatus("");
        setExecutionError(null);
        setDebugResult(null);
      }
      setDebugOpen(true);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setLogs((prev) => [...prev, `[SAVE] failed: ${msg}`]);
      alert(`保存失败：${msg}`);
    } finally {
      setSaving(false);
    }
  }

  function appendNodeAt(template: { type: NodeType; label: string }, x: number, y: number) {
    const id = `${template.type}-${crypto.randomUUID().slice(0, 8)}`;
    const llmProvider =
      template.type === "llm"
        ? template.label === "通义千问"
          ? ("dashscope" as const)
          : template.label === "DeepSeek"
            ? ("deepseek" as const)
            : undefined
        : undefined;
    const meta: WorkflowNodeMeta = {
      id,
      type: template.type,
      data: {
        name: template.label,
        ...(llmProvider ? { provider: llmProvider } : {})
      }
    };
    setNodes((prev) => [
      ...prev,
      {
        id,
        type: "workflowNode",
        position: { x, y },
        data: { label: template.label, nodeType: template.type },
        draggable: true,
        meta
      }
    ]);
  }

  function addOutputParam(nodeId: string) {
    const next: OutputParam = {
      id: crypto.randomUUID().slice(0, 8),
      name: "",
      type: "input",
      value: ""
    };
    setOutputConfigs((prev) => ({ ...prev, [nodeId]: [...(prev[nodeId] || []), next] }));
  }

  function updateOutputParam(nodeId: string, paramId: string, patch: Partial<OutputParam>) {
    setOutputConfigs((prev) => ({
      ...prev,
      [nodeId]: (prev[nodeId] || []).map((item) => (item.id === paramId ? { ...item, ...patch } : item))
    }));
  }

  function removeOutputParam(nodeId: string, paramId: string) {
    setOutputConfigs((prev) => ({
      ...prev,
      [nodeId]: (prev[nodeId] || []).filter((item) => item.id !== paramId)
    }));
  }

  function addDeepSeekInputParam() {
    setDeepSeekConfig((prev) => ({
      ...prev,
      inputParams: [...prev.inputParams, { id: crypto.randomUUID().slice(0, 8), name: "", type: "input", value: "" }]
    }));
  }

  function updateDeepSeekInputParam(paramId: string, patch: Partial<DeepSeekInputParam>) {
    setDeepSeekConfig((prev) => ({
      ...prev,
      inputParams: prev.inputParams.map((item) => (item.id === paramId ? { ...item, ...patch } : item))
    }));
  }

  function removeDeepSeekInputParam(paramId: string) {
    setDeepSeekConfig((prev) => ({
      ...prev,
      inputParams: prev.inputParams.filter((item) => item.id !== paramId)
    }));
  }

  function addDeepSeekOutputParam() {
    setDeepSeekConfig((prev) => ({
      ...prev,
      outputParams: [
        ...prev.outputParams,
        { id: crypto.randomUUID().slice(0, 8), name: "", valueType: "string", description: "" }
      ]
    }));
  }

  function updateDeepSeekOutputParam(paramId: string, patch: Partial<DeepSeekOutputParam>) {
    setDeepSeekConfig((prev) => ({
      ...prev,
      outputParams: prev.outputParams.map((item) => (item.id === paramId ? { ...item, ...patch } : item))
    }));
  }

  function removeDeepSeekOutputParam(paramId: string) {
    setDeepSeekConfig((prev) => ({
      ...prev,
      outputParams: prev.outputParams.filter((item) => item.id !== paramId)
    }));
  }

  function addQwenInputParam() {
    setQwenConfig((prev) => ({
      ...prev,
      inputParams: [...prev.inputParams, { id: crypto.randomUUID().slice(0, 8), name: "", type: "input", value: "" }]
    }));
  }

  function updateQwenInputParam(paramId: string, patch: Partial<DeepSeekInputParam>) {
    setQwenConfig((prev) => ({
      ...prev,
      inputParams: prev.inputParams.map((item) => (item.id === paramId ? { ...item, ...patch } : item))
    }));
  }

  function removeQwenInputParam(paramId: string) {
    setQwenConfig((prev) => ({
      ...prev,
      inputParams: prev.inputParams.filter((item) => item.id !== paramId)
    }));
  }

  function addQwenOutputParam() {
    setQwenConfig((prev) => ({
      ...prev,
      outputParams: [
        ...prev.outputParams,
        { id: crypto.randomUUID().slice(0, 8), name: "", valueType: "string", description: "" }
      ]
    }));
  }

  function updateQwenOutputParam(paramId: string, patch: Partial<DeepSeekOutputParam>) {
    setQwenConfig((prev) => ({
      ...prev,
      outputParams: prev.outputParams.map((item) => (item.id === paramId ? { ...item, ...patch } : item))
    }));
  }

  function removeQwenOutputParam(paramId: string) {
    setQwenConfig((prev) => ({
      ...prev,
      outputParams: prev.outputParams.filter((item) => item.id !== paramId)
    }));
  }

  function deleteNode(nodeId: string) {
    // 同时移除节点本身、与之相连的边，以及该节点的输出配置状态
    if (!confirm("确定删除该节点吗？同时会删除所有连接的边。")) return;
    setNodes((prev) => prev.filter((n) => n.id !== nodeId));
    setEdges((prev) => prev.filter((e) => e.source !== nodeId && e.target !== nodeId));
    setOutputConfigs((prev) => {
      const next = { ...prev };
      delete next[nodeId];
      return next;
    });
    setOutputTemplates((prev) => {
      const next = { ...prev };
      delete next[nodeId];
      return next;
    });
    setSelectedNode(null);
  }

  function insertTemplateVar(nodeId: string, name: string) {
    if (!name) return;
    setOutputTemplates((prev) => ({ ...prev, [nodeId]: `${prev[nodeId] ?? ""}{{${name}}}` }));
  }

  function handleDragStart(event: DragEvent<HTMLButtonElement>, nodeTemplate: { type: NodeType; label: string }) {
    event.dataTransfer.setData("application/walnut-node", JSON.stringify(nodeTemplate));
    event.dataTransfer.effectAllowed = "move";
  }

  function handleDragOver(event: DragEvent<HTMLDivElement>) {
    // 仅当是“节点库 -> 画布”的自定义拖拽数据时，才阻止默认行为；
    // 否则会与 ReactFlow 内部的节点拖拽产生冲突，导致节点无法移动。
    const types = Array.from(event.dataTransfer.types);
    if (!types.includes("application/walnut-node")) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    const raw = event.dataTransfer.getData("application/walnut-node");
    if (!raw || !flowRef.current) return;
    event.preventDefault();
    const template = JSON.parse(raw) as { type: NodeType; label: string };
    const bounds = event.currentTarget.getBoundingClientRect();
    const position = flowRef.current.project({
      x: event.clientX - bounds.left,
      y: event.clientY - bounds.top
    });
    appendNodeAt(template, position.x, position.y);
  }

  function startResizeDebugDrawer(event: React.MouseEvent<HTMLDivElement>) {
    event.preventDefault();
    setDebugOpen(true);
    debugResizeStartRef.current = { startY: event.clientY, startHeight: debugDrawerHeight };
    setIsResizingDebugDrawer(true);
  }

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">
          <strong>PaiAgent</strong>
          <span className="workspace">qoder5</span>
        </div>
        <div className="top-actions">
            <button className="btn ghost" onClick={handleNewWorkflow}>
              ＋ 新建
            </button>
            <button
              className="btn ghost"
              onClick={() => {
                setLoadModalOpen(true);
                void loadWorkflowsList();
              }}
            >
              📂 加载
            </button>
            <button className="btn primary" onClick={() => void handleSaveWorkflow()} disabled={saving}>
              {saving ? "保存中..." : "💾 保存"}
            </button>
            <button
              className="btn primary"
              onClick={() => {
                setDebugOpen(true);
                void handleDebug();
              }}
              disabled={loading}
            >
              🧪 调试
            </button>
          <span className="user">admin</span>
        </div>
      </header>

      <div className="layout">
        <aside className="panel left">
          <h3>节点库</h3>
          {baseNodes.map((n) => (
            <button key={n.type} className="node-btn" draggable onDragStart={(e) => handleDragStart(e, n)}>
              <span className="node-icon">{n.icon}</span>
              <span>{n.label}</span>
            </button>
          ))}
          <p className="group-title">📁 大模型节点</p>
          {llmNodes.map((n) => (
            <button key={n.label} className="node-btn" draggable onDragStart={(e) => handleDragStart(e, n)}>
              <span className="node-icon">{n.icon}</span>
              <span>{n.label}</span>
            </button>
          ))}
          <p className="group-title">🔧 工具节点</p>
          {toolNodes.map((n) => (
            <button key={n.label} className="node-btn" draggable onDragStart={(e) => handleDragStart(e, n)}>
              <span className="node-icon">{n.icon}</span>
              <span>{n.label}</span>
            </button>
          ))}
          <p className="left-tip">💡 拖拽节点到画布中使用</p>
        </aside>

        <main className="canvas-wrap" onDragOver={handleDragOver} onDrop={handleDrop}>
          <ReactFlow
            nodeTypes={NODE_TYPES}
            nodes={nodes}
            edges={edges}
            fitView
            nodesDraggable={true}
            nodesConnectable={true}
            connectOnClick={false}
            defaultEdgeOptions={{
              markerEnd: { type: MarkerType.ArrowClosed, color: "#6b7280" }
            }}
            onConnect={(params) => {
              const { source, target } = params;
              if (!source || !target) return;
              const edgeId = `e-${source}-${target}-${Date.now()}`;
              setEdges((prev) => [
                ...prev,
                {
                  id: edgeId,
                  source,
                  target,
                  animated: true,
                  markerEnd: { type: MarkerType.ArrowClosed, color: "#6b7280" }
                }
              ]);
            }}
            onInit={(instance) => {
              flowRef.current = instance;
            }}
            onNodesChange={(changes) => {
              // ReactFlow 拖拽节点会产生位置变更；这里将其回写到受控状态，避免拖拽后回弹。
              setNodes((prev) => {
                const next = applyNodeChanges(changes, prev);
                // applyNodeChanges 只保证 ReactFlow 标准字段更新，
                // 我们的节点自定义字段 `meta` 需要在更新后继续保留。
                const metaById = new Map(prev.map((n) => [n.id, n.meta]));
                return next.map((n) => ({
                  ...n,
                  meta: metaById.get(n.id) ?? (n as FlowNode).meta
                })) as FlowNode[];
              });
            }}
            onNodeClick={(event, node) => {
              event.stopPropagation();
              const anyNode = node as unknown as { meta?: WorkflowNodeMeta; data: any };
              if (anyNode.meta) {
                setSelectedNode(anyNode.meta);
                return;
              }
              const nodeType = (anyNode.data?.nodeType ?? "input") as NodeType;
              setSelectedNode({
                id: node.id,
                type: nodeType,
                data: { name: anyNode.data?.label }
              });
            }}
            onPaneClick={(event) => {
              const target = event?.target as unknown;
              if (target instanceof HTMLElement && target.closest(".workflow-node")) {
                return;
              }
              setSelectedNode(null);
            }}
          >
            <Background gap={22} size={1} />
            <Controls />
          </ReactFlow>
        </main>

        <aside className="panel right">
          <h3>节点配置</h3>
          {selectedNode ? (
            <div className="config-stack">
              <label className="config-label">节点 ID</label>
              <div className="config-input">{selectedNode.id}</div>
              <label className="config-label">节点类型</label>
              <div className="config-input">{selectedNode.type}</div>
              <button className="remove-btn" onClick={() => deleteNode(selectedNode.id)}>
                删除此节点
              </button>
              {selectedNode.type === "input" ? (
                <>
                  <label className="config-label">变量名</label>
                  <div className="config-input">user_input</div>
                  <label className="config-label">变量类型</label>
                  <div className="config-input">String</div>
                  <label className="config-label">描述</label>
                  <div className="config-input">用户本轮的输入内容</div>
                  <label className="config-label">是否必要</label>
                  <label className="required-row">
                    <input type="checkbox" checked readOnly />
                    <span>必要</span>
                  </label>
                  <button className="save-config-btn">保存配置</button>
                </>
              ) : selectedNode.type === "output" ? (
                <>
                  <label className="config-label">输出配置</label>
                  <button className="add-param-btn" onClick={() => addOutputParam(selectedNode.id)}>
                    ＋ 添加
                  </button>
                  <div className="output-table">
                    {selectedOutputParams.length === 0 ? (
                      <div className="hint">暂无参数，点击“添加”创建</div>
                    ) : (
                      selectedOutputParams.map((item) => (
                        <div className="output-row" key={item.id}>
                          <input
                            className="config-input inline-input"
                            placeholder="参数名"
                            value={item.name}
                            onChange={(e) => updateOutputParam(selectedNode.id, item.id, { name: e.target.value })}
                          />
                          <select
                            className="config-input inline-input"
                            value={item.type}
                            onChange={(e) =>
                              updateOutputParam(selectedNode.id, item.id, { type: e.target.value as OutputParamType, value: "" })
                            }
                          >
                            <option value="input">输入</option>
                            <option value="reference">引用</option>
                          </select>
                          {item.type === "input" ? (
                            <input
                              className="config-input inline-input"
                              placeholder="请输入值"
                              value={item.value}
                              onChange={(e) => updateOutputParam(selectedNode.id, item.id, { value: e.target.value })}
                            />
                          ) : (
                            <select
                              className="config-input inline-input"
                              value={item.value}
                              onChange={(e) => updateOutputParam(selectedNode.id, item.id, { value: e.target.value })}
                            >
                              <option value="">选择引用</option>
                              {referenceOptions.map((option) => (
                                <option key={option.value} value={option.value}>
                                  {option.label}
                                </option>
                              ))}
                            </select>
                          )}
                          <button className="remove-btn" onClick={() => removeOutputParam(selectedNode.id, item.id)}>
                            移除
                          </button>
                        </div>
                      ))
                    )}
                  </div>
                  <label className="config-label">回答内容配置</label>
                  <div className="template-help">
                    {selectedOutputParams.map((item) => (
                      <button key={item.id} className="chip" onClick={() => insertTemplateVar(selectedNode.id, item.name)}>
                        {`{{${item.name || "参数名"}}}`}
                      </button>
                    ))}
                  </div>
                  <textarea
                    className="config-area"
                    value={selectedOutputTemplate}
                    onChange={(e) => setOutputTemplates((prev) => ({ ...prev, [selectedNode.id]: e.target.value }))}
                  />
                  <button className="save-config-btn">保存配置</button>
                </>
              ) : isQwenNode ? (
                <>
                  <p className="hint">通义千问使用阿里云 DashScope 的 OpenAI 兼容模式（与 OpenAI Chat Completions 一致）</p>
                  <label className="config-label">API 根地址（必填）</label>
                  <input
                    className="config-input"
                    placeholder={DEFAULT_QWEN_BASE_URL}
                    value={qwenConfig.baseUrl}
                    onChange={(e) => setQwenConfig((prev) => ({ ...prev, baseUrl: e.target.value }))}
                  />
                  <label className="config-label">DashScope API Key</label>
                  <input
                    type="password"
                    className="config-input"
                    placeholder="在阿里云百炼控制台创建 API Key"
                    value={qwenConfig.apiKey}
                    onChange={(e) => setQwenConfig((prev) => ({ ...prev, apiKey: e.target.value }))}
                  />
                  <label className="config-label">模型名称</label>
                  <select
                    className="config-input"
                    value={(QWEN_MODEL_PRESETS as readonly string[]).includes(qwenConfig.model) ? qwenConfig.model : "custom"}
                    onChange={(e) => {
                      const v = e.target.value;
                      if (v === "custom") {
                        setQwenConfig((prev) => ({
                          ...prev,
                          model: (QWEN_MODEL_PRESETS as readonly string[]).includes(prev.model) ? "" : prev.model
                        }));
                      } else {
                        setQwenConfig((prev) => ({ ...prev, model: v }));
                      }
                    }}
                  >
                    {QWEN_MODEL_PRESETS.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                    <option value="custom">自定义模型 ID…</option>
                  </select>
                  {(!(QWEN_MODEL_PRESETS as readonly string[]).includes(qwenConfig.model) || qwenConfig.model === "") && (
                    <input
                      className="config-input"
                      style={{ marginTop: 8 }}
                      placeholder="与控制台模型列表一致，例如 qwen3-max"
                      value={qwenConfig.model}
                      onChange={(e) => setQwenConfig((prev) => ({ ...prev, model: e.target.value }))}
                    />
                  )}
                  <label className="config-label">输入参数配置</label>
                  <button className="add-param-btn" onClick={addQwenInputParam}>
                    ＋ 添加
                  </button>
                  <div className="output-table">
                    {qwenConfig.inputParams.length === 0 ? (
                      <div className="hint">暂无输入参数，点击“添加”创建</div>
                    ) : (
                      qwenConfig.inputParams.map((item) => (
                        <div className="output-row" key={item.id}>
                          <input
                            className="config-input inline-input"
                            placeholder="参数名"
                            value={item.name}
                            onChange={(e) => updateQwenInputParam(item.id, { name: e.target.value })}
                          />
                          <select
                            className="config-input inline-input"
                            value={item.type}
                            onChange={(e) =>
                              updateQwenInputParam(item.id, { type: e.target.value as DeepSeekInputParamType, value: "" })
                            }
                          >
                            <option value="input">输入</option>
                            <option value="reference">引用</option>
                          </select>
                          {item.type === "input" ? (
                            <input
                              className="config-input inline-input"
                              placeholder="请输入值"
                              value={item.value}
                              onChange={(e) => updateQwenInputParam(item.id, { value: e.target.value })}
                            />
                          ) : (
                            <select
                              className="config-input inline-input"
                              value={item.value}
                              onChange={(e) => updateQwenInputParam(item.id, { value: e.target.value })}
                            >
                              <option value="">选择引用</option>
                              {deepSeekReferenceOptions.map((option) => (
                                <option key={option.value} value={option.value}>
                                  {option.label}
                                </option>
                              ))}
                            </select>
                          )}
                          <button className="remove-btn" onClick={() => removeQwenInputParam(item.id)}>
                            移除
                          </button>
                        </div>
                      ))
                    )}
                  </div>
                  <label className="config-label">提示词模板</label>
                  <textarea
                    className="config-area"
                    placeholder={"# 角色\n你是一位专业的广播节目编辑...\n# 原始内容：{{input}}"}
                    value={qwenConfig.promptTemplate}
                    onChange={(e) => setQwenConfig((prev) => ({ ...prev, promptTemplate: e.target.value }))}
                  />
                  <div className="template-help">
                    {qwenConfig.inputParams.map((item) => (
                      <button
                        key={item.id}
                        className="chip"
                        onClick={() =>
                          setQwenConfig((prev) => ({
                            ...prev,
                            promptTemplate: `${prev.promptTemplate}{{${item.name || "参数名"}}}`
                          }))
                        }
                      >
                        {`{{${item.name || "参数名"}}}`}
                      </button>
                    ))}
                  </div>
                  <label className="config-label">输出参数配置</label>
                  <button className="add-param-btn" onClick={addQwenOutputParam}>
                    ＋ 添加
                  </button>
                  <div className="output-table">
                    {qwenConfig.outputParams.length === 0 ? (
                      <div className="hint">暂无输出参数，点击“添加”创建</div>
                    ) : (
                      qwenConfig.outputParams.map((item) => (
                        <div className="output-row" key={item.id}>
                          <input
                            className="config-input inline-input"
                            placeholder="变量名"
                            value={item.name}
                            onChange={(e) => updateQwenOutputParam(item.id, { name: e.target.value })}
                          />
                          <select
                            className="config-input inline-input"
                            value={item.valueType}
                            onChange={(e) => updateQwenOutputParam(item.id, { valueType: e.target.value as "string" })}
                          >
                            <option value="string">string</option>
                          </select>
                          <input
                            className="config-input inline-input"
                            placeholder="描述（可为空）"
                            value={item.description}
                            onChange={(e) => updateQwenOutputParam(item.id, { description: e.target.value })}
                          />
                          <button className="remove-btn" onClick={() => removeQwenOutputParam(item.id)}>
                            移除
                          </button>
                        </div>
                      ))
                    )}
                  </div>
                  <label className="config-label">温度（temperature）</label>
                  <input
                    type="range"
                    min={0}
                    max={2}
                    step={0.1}
                    value={qwenConfig.temperature}
                    onChange={(e) => setQwenConfig((prev) => ({ ...prev, temperature: Number(e.target.value) }))}
                  />
                  <div className="temp-tip">
                    当前值：{qwenConfig.temperature.toFixed(1)}。温度越低越严谨，越高越发散。
                  </div>
                  <button className="save-config-btn" onClick={saveQwenConfig}>
                    保存配置
                  </button>
                </>
              ) : isDeepSeekNode ? (
                <>
                  <label className="config-label">模型接口地址（必填）</label>
                  <input
                    className="config-input"
                    placeholder="例如: https://api.deepseek.com/v1"
                    value={deepSeekConfig.baseUrl}
                    onChange={(e) => setDeepSeekConfig((prev) => ({ ...prev, baseUrl: e.target.value }))}
                  />
                  <label className="config-label">API 密钥</label>
                  <input
                    type="password"
                    className="config-input"
                    placeholder="请输入 API Key"
                    value={deepSeekConfig.apiKey}
                    onChange={(e) => setDeepSeekConfig((prev) => ({ ...prev, apiKey: e.target.value }))}
                  />
                  <label className="config-label">模型名称</label>
                  <input className="config-input" value="deepseek-chat" readOnly />
                  <label className="config-label">输入参数配置</label>
                  <button className="add-param-btn" onClick={addDeepSeekInputParam}>
                    ＋ 添加
                  </button>
                  <div className="output-table">
                    {deepSeekConfig.inputParams.length === 0 ? (
                      <div className="hint">暂无输入参数，点击“添加”创建</div>
                    ) : (
                      deepSeekConfig.inputParams.map((item) => (
                        <div className="output-row" key={item.id}>
                          <input
                            className="config-input inline-input"
                            placeholder="参数名"
                            value={item.name}
                            onChange={(e) => updateDeepSeekInputParam(item.id, { name: e.target.value })}
                          />
                          <select
                            className="config-input inline-input"
                            value={item.type}
                            onChange={(e) =>
                              updateDeepSeekInputParam(item.id, { type: e.target.value as DeepSeekInputParamType, value: "" })
                            }
                          >
                            <option value="input">输入</option>
                            <option value="reference">引用</option>
                          </select>
                          {item.type === "input" ? (
                            <input
                              className="config-input inline-input"
                              placeholder="请输入值"
                              value={item.value}
                              onChange={(e) => updateDeepSeekInputParam(item.id, { value: e.target.value })}
                            />
                          ) : (
                            <select
                              className="config-input inline-input"
                              value={item.value}
                              onChange={(e) => updateDeepSeekInputParam(item.id, { value: e.target.value })}
                            >
                              <option value="">选择前置节点引用</option>
                              {deepSeekReferenceOptions.map((option) => (
                                <option key={option.value} value={option.value}>
                                  {option.label}
                                </option>
                              ))}
                            </select>
                          )}
                          <button className="remove-btn" onClick={() => removeDeepSeekInputParam(item.id)}>
                            移除
                          </button>
                        </div>
                      ))
                    )}
                  </div>
                  <label className="config-label">用户提示词（支持 {"{{input}}" }）</label>
                  <textarea
                    className="config-area"
                    placeholder={"# 角色\n你是一位专业的广播节目编辑...\n# 原始内容：{{input}}"}
                    value={deepSeekConfig.promptTemplate}
                    onChange={(e) => setDeepSeekConfig((prev) => ({ ...prev, promptTemplate: e.target.value }))}
                  />
                  <div className="template-help">
                    {deepSeekConfig.inputParams.map((item) => (
                      <button key={item.id} className="chip" onClick={() => setDeepSeekConfig((prev) => ({ ...prev, promptTemplate: `${prev.promptTemplate}{{${item.name || "参数名"}}}` }))}>
                        {`{{${item.name || "参数名"}}}`}
                      </button>
                    ))}
                  </div>
                  <label className="config-label">输出参数配置</label>
                  <button className="add-param-btn" onClick={addDeepSeekOutputParam}>
                    ＋ 添加
                  </button>
                  <div className="output-table">
                    {deepSeekConfig.outputParams.length === 0 ? (
                      <div className="hint">暂无输出参数，点击“添加”创建</div>
                    ) : (
                      deepSeekConfig.outputParams.map((item) => (
                        <div className="output-row" key={item.id}>
                          <input
                            className="config-input inline-input"
                            placeholder="变量名"
                            value={item.name}
                            onChange={(e) => updateDeepSeekOutputParam(item.id, { name: e.target.value })}
                          />
                          <select
                            className="config-input inline-input"
                            value={item.valueType}
                            onChange={(e) => updateDeepSeekOutputParam(item.id, { valueType: e.target.value as "string" })}
                          >
                            <option value="string">string</option>
                          </select>
                          <input
                            className="config-input inline-input"
                            placeholder="描述（可为空）"
                            value={item.description}
                            onChange={(e) => updateDeepSeekOutputParam(item.id, { description: e.target.value })}
                          />
                          <button className="remove-btn" onClick={() => removeDeepSeekOutputParam(item.id)}>
                            移除
                          </button>
                        </div>
                      ))
                    )}
                  </div>
                  <label className="config-label">温度（temperature）</label>
                  <input
                    type="range"
                    min={0}
                    max={2}
                    step={0.1}
                    value={deepSeekConfig.temperature}
                    onChange={(e) => setDeepSeekConfig((prev) => ({ ...prev, temperature: Number(e.target.value) }))}
                  />
                  <div className="temp-tip">
                    当前值：{deepSeekConfig.temperature.toFixed(1)}。温度越低越严谨，越高越发散。
                  </div>
                  <button className="save-config-btn" onClick={saveDeepSeekConfig}>
                    保存配置
                  </button>
                </>
              ) : (
                <>
                  <label className="config-label">输出配置</label>
                  <div className="config-row">
                    <span className="chip">output</span>
                    <span className="chip">引用</span>
                    <span className="chip">{selectedNode.data?.name || "字段"}</span>
                  </div>
                  <label className="config-label">回答内容配置</label>
                  <textarea className="config-area" value="{{output}}" readOnly />
                  <button className="save-config-btn">保存配置</button>
                </>
              )}
            </div>
          ) : (
            <p className="hint">点击画布节点查看配置</p>
          )}
        </aside>
      </div>

      {loadModalOpen && (
        <div className="modal-overlay" onClick={() => setLoadModalOpen(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <strong>加载工作流</strong>
              <button className="modal-close" onClick={() => setLoadModalOpen(false)}>
                ×
              </button>
            </div>
            <div className="modal-body">
              {workflowsLoading ? (
                <div className="hint">加载中...</div>
              ) : workflowsList.length === 0 ? (
                <div className="hint">暂无工作流</div>
              ) : (
                workflowsList.map((w) => (
                  <div key={w.id} className="workflow-row">
                    <div className="workflow-row-left">
                      <div className="workflow-name">{w.name}</div>
                      <div className="workflow-sub">
                        {w.draft ? "草稿" : ""}
                        {w.published ? (w.draft ? " / 发布" : "发布") : ""}
                      </div>
                    </div>
                    <div className="workflow-actions">
                      <button className="workflow-load-btn" onClick={() => void handleLoadWorkflow(w.id)}>
                        加载
                      </button>
                      <button className="workflow-delete-btn" onClick={() => void handleDeleteWorkflow(w.id)} disabled={workflowsLoading}>
                        删除
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}

      <section
        className={`debug-drawer ${debugOpen ? "open" : ""} ${isResizingDebugDrawer ? "resizing" : ""}`}
        style={{ height: debugOpen ? debugDrawerHeight : DEBUG_DRAWER_MIN_HEIGHT }}
      >
        <div
          className="debug-resize-handle"
          onMouseDown={startResizeDebugDrawer}
          title="拖动调节调试框高度"
        />
        <div className="debug-header">
          <strong>调试抽屉</strong>
          <button className="toggle-btn" onClick={() => setDebugOpen((v) => !v)}>
            {debugOpen ? "收起" : "展开"}
          </button>
        </div>
        {debugOpen && (
          <div className="debug-body">
            <textarea
              value={debugInput}
              onChange={(e) => setDebugInput(e.target.value)}
              placeholder="输入测试文本..."
            />
            <button className="run-btn" onClick={handleDebug} disabled={loading}>
              {loading ? "调试中..." : "开始调试"}
            </button>
            <div className="result-box">
              <div className="exec-head">
                <div>执行状态: {executionStatus || "-"}</div>
                {executionError ? <div className="exec-error">错误: {executionError}</div> : null}
              </div>

              <div className="exec-nodes-head">
                节点执行状态（{liveNodeResults.length}）
              </div>
              <div className="exec-nodes">
                {liveNodeResults.length ? (
                  liveNodeResults.map((n) => (
                    <div key={n.nodeId} className="exec-node-row">
                      <span className={`exec-node-dot ${n.status === "SUCCESS" ? "ok" : n.status === "FAILED" ? "bad" : ""}`}>
                        {n.status === "SUCCESS" ? "✓" : n.status === "FAILED" ? "×" : n.status === "RUNNING" ? "..." : "?"}
                      </span>
                      <span className="exec-node-type">{n.nodeType}</span>
                      <span className="exec-node-status">{n.status}</span>
                      {n.durationMs != null ? <span className="exec-node-dur">{n.durationMs}ms</span> : null}
                    </div>
                  ))
                ) : (
                  <div className="hint">暂无节点结果</div>
                )}
              </div>

              <div className="exec-io">
                <div className="exec-io-row">
                  <div>输入数据:</div>
                  <pre className="io-pre">{latestExecution?.inputText || "-"}</pre>
                </div>
                <div className="exec-io-row">
                  <div>输出数据:</div>
                  <pre className="io-pre">{debugResult?.text || latestExecution?.outputText || "-"}</pre>
                </div>
              </div>

              <div className="exec-audio">
                音频状态: {debugResult?.audioBase64 ? "已返回音频并自动播放" : "未返回音频"}
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
