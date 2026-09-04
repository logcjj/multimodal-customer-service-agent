export interface Provider {
  id: string;
  name: string;
  capabilities: string[];
  accent: string;
  model_presets?: Record<string, ProviderModelPreset[]>;
}

export interface ProviderModelPreset {
  name: string;
  description: string;
}

export interface ModelConfig {
  id: string;
  name: string;
  kind: string;
  provider: string;
  base_url: string;
  secret_configured: boolean;
  secret_hint: string | null;
  capabilities: string[];
  enabled: boolean;
  is_default: boolean;
  health: 'untested' | 'healthy' | 'unhealthy';
  latency_ms: number | null;
}

export interface AgentDefinition {
  id: string;
  name: string;
  short_name: string;
  description: string;
  execution_mode: string;
  status: string;
  accent: string;
  skills: string[];
  tools: string[];
}

export interface SkillDefinition {
  id: string;
  name: string;
  owner: string;
  version: string;
  status: string;
  description: string;
}

export interface ToolDefinition {
  id: string;
  name: string;
  risk_level: string;
  requires_confirmation: boolean;
  timeout_ms: number;
  idempotent: boolean;
}

export interface McpServer {
  id: string;
  name: string;
  description: string;
  mode: string;
  transport: string;
  status: string;
  tools: string[];
  resources: string[];
}

export interface Evidence {
  evidence_id: string;
  source_type: string;
  title: string;
  text: string;
  product?: string | null;
  dataset_id?: string | null;
  document_id?: string | null;
  file_id?: string | null;
  document_name?: string | null;
  document_mime_type?: string | null;
  document_version?: string | null;
  section_id?: string | null;
  parent_id?: string | null;
  child_ids: string[];
  image_chunk_ids: string[];
  chapter_title?: string | null;
  page_start?: number | null;
  page_end?: number | null;
  locator_label?: string | null;
  asset_ids: string[];
  score?: number | null;
  score_breakdown: Record<string, number>;
  retrieval_stage?: string | null;
  evidence_confidence?: number | null;
}

export interface AgentStep {
  agent_id: string;
  label: string;
  status: string;
  latency_ms: number;
  summary: string;
}

export interface AgentTrace {
  request_id: string;
  session_id: string;
  route: string;
  selected_agents: string[];
  steps: AgentStep[];
  spans: Array<{
    span_id: string;
    name: string;
    status: string;
    latency_ms: number;
    input_summary: string;
    output_summary: string;
    attributes: Record<string, unknown>;
  }>;
  fallback_reason: string | null;
  total_latency_ms: number;
  created_at?: string;
}

export interface ClarificationRequest {
  case_id: string;
  field: string;
  question: string;
  round: number;
  max_rounds: number;
  accepted_input_types: string[];
}

export type FinalRoute =
  | 'technical_knowledge'
  | 'customer_service'
  | 'mixed'
  | 'evidence_clarification'
  | 'general_llm'
  | 'safe_handoff'
  | 'general_unavailable';

export interface RoutingDecision {
  initial_route: string;
  final_route: FinalRoute;
  route_label: string;
  route_reason: string;
  coverage_status:
    | 'covered'
    | 'clarifiable'
    | 'unsafe_uncovered'
    | 'general_allowed'
    | 'general_unavailable';
  knowledge_covered: boolean;
  risk_level: 'low' | 'medium' | 'high';
  clarification: ClarificationRequest | null;
}

export interface AgentResponse {
  request_id: string;
  session_id: string;
  answer: string;
  route: string;
  citations: Evidence[];
  assets: string[];
  verification: {
    passed: boolean;
    action: string;
    confidence: number;
    issues: Array<{ code: string; message: string; severity: string }>;
  };
  trace: AgentTrace;
  used_legacy: boolean;
  routing?: RoutingDecision | null;
  timestamp: number;
}

export interface ConversationSummary {
  id: string;
  owner_id: string;
  title: string;
  message_count: number;
  last_message_preview: string;
  last_route: string | null;
  created_at: string;
  updated_at: string;
}

export interface ConversationTurn {
  id: string;
  conversation_id: string;
  ordinal: number;
  request_id: string;
  user_text: string;
  attachment_metadata: Array<Record<string, unknown>>;
  assistant_text: string;
  response: AgentResponse | null;
  status: 'running' | 'completed' | 'failed';
  error_code: string | null;
  initial_route: string | null;
  final_route: string | null;
  route_reason: string | null;
  coverage_status: string | null;
  created_at: string;
  completed_at: string | null;
}

export interface ConversationDetail extends ConversationSummary {
  turns: ConversationTurn[];
  state: {
    slots: Record<string, unknown>;
    rolling_summary: string;
    summary_through_ordinal: number;
    pending_clarification: Record<string, unknown> | null;
    clarification_round: number;
    memory_version: number;
    updated_at: string | null;
  } | null;
}

export interface SessionMemory {
  session_id: string;
  user_id: string | null;
  turn_count: number;
  last_question: string;
  products: string[];
  model_codes: string[];
  intent: string;
  answer_summary: string;
  evidence_refs: Array<{
    evidence_id: string;
    source_type: string;
    title: string;
    dataset_id: string | null;
    document_id: string | null;
    parent_id: string | null;
    image_chunk_ids: string[];
  }>;
  visual_context: {
    image_hashes: string[];
    ocr_text: string;
    detected_codes: string[];
    detected_numbers: string[];
    detected_product: string | null;
    detected_components: string[];
    visible_objects: string[];
    visual_summary: string;
    provider_status: Record<string, string>;
    confidence: number;
  } | null;
  missing_information: string[];
  risk_state: string;
  created_at: string;
  updated_at: string;
  expires_at: string;
}

export interface RuntimeReadiness {
  status: string;
  rollout_mode: string;
  legacy_available: boolean;
  model_registry: string;
  trace_store: string;
  llm_configured: boolean;
  llm_model: string | null;
  vlm_configured: boolean;
  embedding_configured: boolean;
  rerank_configured: boolean;
  ocr_configured: boolean;
  dynamic_routing: string;
  conversation_history: string;
  layered_memory: string;
  general_agent: string;
}

export interface RuntimeCapabilities {
  multi_agent: boolean;
  vision: boolean;
  skills: boolean;
  mcp: boolean;
  memory: boolean;
  trace: boolean;
  legacy_fallback: boolean;
  stream_mode: string;
  agent_count: number;
  skill_count: number;
  tool_count: number;
  dynamic_routing: string;
  conversation_history: string;
  layered_memory: string;
  general_agent: string;
}

export interface RuntimeEvent {
  sequence: number;
  type: string;
  agent_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  label: string;
  summary: string;
  payload: {
    selected_agents?: string[];
    route?: string;
    llm_configured?: boolean;
    llm_model?: string | null;
    llm_generated?: boolean;
    model_used?: string | null;
    evidence_count?: number;
    delta?: string;
    answer?: string;
    response?: AgentResponse;
    [key: string]: unknown;
  };
}

export interface FileAsset {
  id: string;
  original_name: string;
  content_hash: string;
  mime_type: string;
  size_bytes: number;
  storage_path: string;
  status: string;
  created_at: string;
}

export interface Dataset {
  id: string;
  name: string;
  description: string;
  parser_profile: string;
  visibility: string;
  published_version: string | null;
  status: string;
  is_system: boolean;
  retrieval_profile_id: string | null;
  document_count: number;
  parent_count: number;
  child_count: number;
  asset_count: number;
  failed_job_count: number;
  created_at: string;
  updated_at: string;
}

export interface KnowledgeDocument {
  id: string;
  dataset_id: string;
  file_id: string;
  original_name: string;
  mime_type: string;
  parser_profile: string;
  enabled: boolean;
  active_version: string | null;
  published_version: string | null;
  latest_job_state: string | null;
  latest_job_progress: number | null;
  created_at: string;
  updated_at: string;
}

export interface ParsingJob {
  id: string;
  document_ref_id: string;
  state: string;
  stage: string;
  progress: number;
  parser_version: string;
  index_version: string | null;
  error_code: string | null;
  error_message: string | null;
  started_at: string | null;
  finished_at: string | null;
}

export interface ParentChunk {
  id: string;
  document_id: string;
  index_version: string;
  title: string;
  heading_path: string[];
  text: string;
  page_start: number;
  page_end: number;
  token_count: number;
  enabled: boolean;
  edited: boolean;
}

export interface ChildChunk {
  id: string;
  parent_id: string;
  document_id: string;
  index_version: string;
  title: string;
  text: string;
  page_start: number;
  page_end: number;
  token_count: number;
  keywords: string[];
  questions: string[];
  tags: string[];
  asset_ids: string[];
  enabled: boolean;
  edited: boolean;
}

export interface ChunkCollection {
  parents: ParentChunk[];
  children: ChildChunk[];
}

export interface ImageChunk {
  id: string;
  dataset_id: string;
  document_id: string;
  index_version: string;
  asset_id: string;
  asset_url: string;
  image_id: string;
  manual_name: string;
  chapter_title: string;
  page_number: number;
  caption: string;
  ocr_text: string;
  visible_text: string[];
  visual_summary: string;
  visual_meaning: string;
  retrieval_text: string;
  search_terms: string[];
  applicable_questions: string[];
  issue_signals: string[];
  related_parent_ids: string[];
  related_child_ids: string[];
  confidence: number;
  content_hash: string;
  embedding_dimension: number;
  enabled: boolean;
}

export interface IndexManifest {
  schema_version: string;
  dataset_id: string;
  index_version: string;
  parent_index_version: string | null;
  built_at: string;
  parser_version: string;
  embedding_model: string | null;
  vector_dimension: number;
  sources: Array<{
    document_id: string;
    file_id: string;
    source_name: string;
    source_sha256: string;
    mime_type: string;
    size_bytes: number;
    parser_fingerprint: string;
    document_version: string | null;
  }>;
  artifacts: Record<
    string,
    { file_name: string; sha256: string; size_bytes: number; row_count: number }
  >;
  counts: Record<string, number>;
  incremental: {
    reused: number;
    added: number;
    updated: number;
    deleted: number;
  };
  validation_status: string;
  evaluation_status: string;
  approval_status: string;
}

export type VectorMapStatus =
  | 'ready'
  | 'building'
  | 'stale'
  | 'failed'
  | 'missing'
  | 'no_published_version'
  | 'no_embeddings';

export interface VectorMapBounds {
  x_min: number;
  x_max: number;
  y_min: number;
  y_max: number;
}

export interface VectorMapUmapMeta {
  n_components: number;
  metric: string;
  n_neighbors: number;
  min_dist: number;
  random_state: number;
  transform_seed: number;
  low_memory?: boolean;
  reducer_available?: boolean;
  small_sample?: boolean;
}

export interface VectorMapMeta {
  dataset_id: string;
  published_version: string;
  embedding_model: string;
  vector_dimension: number;
  point_count: number;
  bounds: VectorMapBounds;
  content_digest: string;
  built_at: string;
  umap: VectorMapUmapMeta;
  projection_version?: string | null;
  created_at?: string | null;
}

export interface VectorMapPoint {
  child_id: string;
  dataset_id: string;
  document_id: string;
  document_name: string;
  title: string;
  excerpt: string;
  page_start: number;
  page_end: number;
  product: string | null;
  x: number;
  y: number;
}

export interface VectorMapError {
  code: string;
  message: string;
  detail?: string | null;
}

export type VectorMapResponse =
  | {
      status: 'ready';
      meta: VectorMapMeta;
      points: VectorMapPoint[];
      message?: string | null;
      error?: null;
    }
  | {
      status: 'building' | 'missing' | 'no_published_version' | 'no_embeddings';
      meta?: Record<string, unknown> | null;
      points?: VectorMapPoint[];
      message?: string | null;
      error?: null;
    }
  | {
      status: 'stale';
      meta?: {
        dataset_id?: string;
        previous_published_version?: string;
        current_published_version?: string;
        [key: string]: unknown;
      } | null;
      points?: VectorMapPoint[];
      message?: string | null;
      error?: null;
    }
  | {
      status: 'failed';
      meta?: Record<string, unknown> | null;
      points?: VectorMapPoint[];
      message?: string | null;
      error: VectorMapError;
    };

export interface RetrievalResult {
  parent_id: string;
  dataset_id: string;
  document_id: string;
  document_version: string;
  title: string;
  text: string;
  product: string | null;
  page_start: number;
  page_end: number;
  asset_ids: string[];
  matched_children: string[];
  scores: {
    lexical: number;
    dense: number;
    rrf: number;
    rerank: number;
    parent: number;
  };
}

export interface RetrievalStageItem {
  id: string;
  title?: string | null;
  score?: number | null;
  dataset_id?: string | null;
  document_id?: string | null;
  child_id?: string | null;
  parent_id?: string | null;
  page_start?: number | null;
  page_end?: number | null;
  rank?: number | null;
}

export interface RetrievalVisualizationPoint {
  x: number;
  y: number;
}

export interface RetrievalVisualizationHit {
  child_id: string;
  rank: number;
  score: number;
}

export interface RetrievalVisualization {
  projection_version: string | null;
  query: RetrievalVisualizationPoint | null;
  dense_top10: RetrievalVisualizationHit[];
  rerank_top10: RetrievalVisualizationHit[];
  rrf_top10: RetrievalVisualizationHit[];
  status?: VectorMapStatus | 'unavailable' | 'query_transform_failed' | null;
  message?: string | null;
}

export interface RetrievalExplanation {
  query: string;
  mode: string;
  results: RetrievalResult[];
  stages: Record<string, RetrievalStageItem[]>;
  rejected_reason: string | null;
  warnings?: string[];
  visualization?: RetrievalVisualization | null;
}

export interface RetrievalProfile {
  id: string;
  name: string;
  lexical_top_k: number;
  dense_top_k: number;
  rrf_k: number;
  rerank_top_k: number;
  final_top_n: number;
  min_score: number;
  min_coverage: number;
  parent_strategy: string;
  empty_response: string;
  created_at: string;
  updated_at: string;
}

export interface EvalCase {
  id: string;
  question: string;
  dataset_ids: string[];
  target_parent_ids: string[];
  reference_answer: string;
  required_facts: string[];
  forbidden_facts: string[];
  image_required: boolean;
  locked: boolean;
  source: string;
  created_at: string;
}

export interface EvalRun {
  id: string;
  candidate_version: string;
  case_ids: string[];
  metrics: Record<string, number>;
  details: Array<Record<string, unknown>>;
  passed: boolean;
  status: string;
  created_at: string;
  approved_at: string | null;
}
