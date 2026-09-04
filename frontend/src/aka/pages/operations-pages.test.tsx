import { api } from '@/aka/api/client';
import { fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { CLIENT_ID_STORAGE_KEY } from '../lib/conversations';
import AgentsPage from './agents-page';
import McpPage from './mcp-page';
import ModelSettingsPage from './model-settings-page';
import SkillsPage from './skills-page';
import TracesPage from './traces-page';

jest.mock('@/aka/api/client', () => ({
  api: {
    agents: jest.fn(),
    skills: jest.fn(),
    tools: jest.fn(),
    mcpServers: jest.fn(),
    traces: jest.fn(),
    readiness: jest.fn(),
    providers: jest.fn(),
    models: jest.fn(),
    createModel: jest.fn(),
    updateModel: jest.fn(),
    setDefaultModel: jest.fn(),
    testModel: jest.fn(),
  },
}));

const mockedApi = api as Record<string, jest.Mock>;

describe('RAGFlow-style operations pages', () => {
  beforeEach(() => {
    window.localStorage.clear();
    window.localStorage.setItem(CLIENT_ID_STORAGE_KEY, 'anon-traces-owner');
    mockedApi.agents.mockResolvedValue([
      {
        id: 'knowledge',
        name: '知识检索智能体',
        short_name: 'Knowledge',
        description: '检索并引用证据',
        execution_mode: 'online',
        status: 'ready',
        accent: '#3fb950',
        skills: ['manual-qa'],
        tools: ['knowledge.search'],
      },
    ]);
    mockedApi.skills.mockResolvedValue([
      {
        id: 'manual-qa',
        name: '说明书问答',
        owner: 'knowledge',
        version: '1.0.0',
        status: 'active',
        description: '基于证据回答',
      },
    ]);
    mockedApi.tools.mockResolvedValue([
      {
        id: 'knowledge.search',
        name: '混合检索',
        risk_level: 'read',
        requires_confirmation: false,
        timeout_ms: 2500,
        idempotent: true,
      },
    ]);
    mockedApi.mcpServers.mockResolvedValue([
      {
        id: 'mcp-1',
        name: 'Customer Service MCP',
        description: '只读知识服务',
        mode: 'read-only',
        transport: 'streamable-http',
        status: 'ready',
        tools: ['knowledge.search'],
        resources: [],
      },
    ]);
    mockedApi.traces.mockResolvedValue([
      {
        request_id: 'req-1',
        session_id: 'session-1',
        route: 'technical',
        selected_agents: ['orchestrator', 'knowledge', 'verifier'],
        steps: [],
        spans: [],
        fallback_reason: null,
        total_latency_ms: 880,
      },
    ]);
    mockedApi.readiness.mockResolvedValue({
      status: 'ready',
      rollout_mode: 'agent_first',
      legacy_available: true,
      model_registry: 'ready',
      trace_store: 'ready',
      llm_configured: true,
      llm_model: 'deepseek-v4-flash',
      vlm_configured: false,
      embedding_configured: false,
      rerank_configured: false,
      ocr_configured: false,
    });
    mockedApi.providers.mockResolvedValue([
      {
        id: 'deepseek',
        name: 'DeepSeek',
        capabilities: ['llm'],
        accent: '#4d6bfe',
      },
    ]);
    mockedApi.models.mockResolvedValue([
      {
        id: 'model-1',
        name: 'deepseek-v4-flash',
        kind: 'llm',
        provider: 'DeepSeek',
        base_url: 'https://api.deepseek.com/v1',
        secret_configured: true,
        secret_hint: '••••1234',
        capabilities: ['llm'],
        enabled: true,
        is_default: true,
        health: 'healthy',
        latency_ms: 670,
      },
    ]);
    mockedApi.testModel.mockResolvedValue({
      health: 'healthy',
      latency_ms: 650,
      message: '连接正常',
    });
  });

  it('renders Agent, Skill, Tool, MCP and Trace metadata', async () => {
    const { rerender } = render(
      <MemoryRouter>
        <AgentsPage />
      </MemoryRouter>,
    );
    expect(await screen.findByText('知识检索智能体')).toBeInTheDocument();

    rerender(
      <MemoryRouter>
        <SkillsPage />
      </MemoryRouter>,
    );
    expect(await screen.findByText('说明书问答')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Tools' }));
    expect(screen.getByText('混合检索')).toBeInTheDocument();

    rerender(
      <MemoryRouter>
        <McpPage />
      </MemoryRouter>,
    );
    expect(await screen.findByText('Customer Service MCP')).toBeInTheDocument();
    expect(screen.getByText('knowledge.search')).toBeInTheDocument();

    rerender(
      <MemoryRouter>
        <TracesPage />
      </MemoryRouter>,
    );
    expect(await screen.findByText('req-1')).toBeInTheDocument();
    expect(screen.getAllByText('880 ms').length).toBeGreaterThan(0);
    expect(screen.getByText('历史回答模型未记录')).toBeInTheDocument();
    expect(screen.queryByText('deepseek-v4-flash')).not.toBeInTheDocument();
    expect(screen.getAllByText('已入队').length).toBeGreaterThan(0);
    expect(mockedApi.traces).toHaveBeenCalledWith('anon-traces-owner');
  });

  it('shows the real default model and tests its health', async () => {
    render(
      <MemoryRouter>
        <ModelSettingsPage />
      </MemoryRouter>,
    );
    expect(await screen.findByText('deepseek-v4-flash')).toBeInTheDocument();
    expect(screen.getByText('健康')).toBeInTheDocument();
    fireEvent.click(
      screen.getByRole('button', { name: '测试 deepseek-v4-flash' }),
    );
    expect(await screen.findByText(/连接正常/)).toBeInTheDocument();
  });

  it('allows a mistakenly classified visual model to be moved into the VLM slot', async () => {
    mockedApi.models.mockResolvedValue([
      {
        id: 'model-1',
        name: 'deepseek-v4-flash',
        kind: 'llm',
        provider: 'DeepSeek',
        base_url: 'https://api.deepseek.com/v1',
        secret_configured: true,
        secret_hint: '•••1234',
        capabilities: ['llm'],
        enabled: true,
        is_default: true,
        health: 'healthy',
        latency_ms: 670,
      },
      {
        id: 'model-vl',
        name: 'qwen3-vl-flash',
        kind: 'llm',
        provider: 'Tongyi-Qianwen',
        base_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        secret_configured: true,
        secret_hint: '•••84ea',
        capabilities: ['llm'],
        enabled: true,
        is_default: false,
        health: 'healthy',
        latency_ms: 254,
      },
    ]);
    mockedApi.updateModel.mockResolvedValue({
      id: 'model-vl',
      name: 'qwen3-vl-flash',
      kind: 'vlm',
      provider: 'Tongyi-Qianwen',
      base_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
      secret_configured: true,
      secret_hint: '•••84ea',
      capabilities: ['vlm'],
      enabled: true,
      is_default: true,
      health: 'untested',
      latency_ms: null,
    });

    render(
      <MemoryRouter>
        <ModelSettingsPage />
      </MemoryRouter>,
    );

    const kindSelect = await screen.findByRole('combobox', {
      name: '设置 qwen3-vl-flash 模型类型',
    });
    fireEvent.change(kindSelect, { target: { value: 'vlm' } });

    expect(mockedApi.updateModel).toHaveBeenCalledWith('model-vl', {
      kind: 'vlm',
    });
    expect(
      await screen.findByText(
        'qwen3-vl-flash 已调整为 VLM，请重新测试连接',
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('option', {
        name: 'qwen3-vl-flash · Tongyi-Qianwen',
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('combobox', { name: '选择默认 VLM 模型' }),
    ).toHaveValue('model-vl');
  });

  it('suggests VLM while adding a model whose name identifies a visual model', async () => {
    mockedApi.providers.mockResolvedValue([
      {
        id: 'tongyi-qianwen',
        name: 'Tongyi-Qianwen',
        capabilities: ['llm', 'embedding', 'rerank', 'vlm'],
        accent: '#615ced',
      },
    ]);
    mockedApi.models.mockResolvedValue([]);
    mockedApi.createModel.mockResolvedValue({
      id: 'model-vl',
      name: 'qwen3-vl-flash',
      kind: 'vlm',
      provider: 'Tongyi-Qianwen',
      base_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
      secret_configured: true,
      secret_hint: '•••84ea',
      capabilities: ['vlm'],
      enabled: true,
      is_default: true,
      health: 'untested',
      latency_ms: null,
    });

    render(
      <MemoryRouter>
        <ModelSettingsPage />
      </MemoryRouter>,
    );

    fireEvent.click(
      await screen.findByRole('button', { name: '添加 Tongyi-Qianwen' }),
    );
    fireEvent.change(screen.getByRole('textbox', { name: '模型名称' }), {
      target: { value: 'qwen3-vl-flash' },
    });

    expect(screen.getByRole('combobox', { name: '模型类型' })).toHaveValue(
      'vlm',
    );
    fireEvent.click(screen.getByRole('button', { name: '保存模型' }));
    expect(mockedApi.createModel).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'vlm', capabilities: ['vlm'] }),
    );
    expect(
      await screen.findByText('qwen3-vl-flash 已添加'),
    ).toBeInTheDocument();
  });

  it('offers OCR, ASR and TTS provider filters and preserves the selected role', async () => {
    mockedApi.providers.mockResolvedValue([
      {
        id: 'tongyi-qianwen',
        name: 'Tongyi-Qianwen',
        capabilities: ['llm', 'embedding', 'rerank', 'vlm', 'ocr', 'asr', 'tts'],
        accent: '#615ced',
        model_presets: {
          rerank: [
            {
              name: 'qwen3-rerank',
              description: '适用于文本语义检索与 RAG 精排',
            },
          ],
          vlm: [
            {
              name: 'qwen3-vl-flash',
              description: 'Qwen3 系列低延迟视觉理解模型',
            },
            {
              name: 'qwen3-vl-plus',
              description: 'Qwen3 系列高能力视觉理解模型',
            },
          ],
          ocr: [
            {
              name: 'qwen3.5-ocr',
              description: '千问 OCR 最新主力版本',
            },
            {
              name: 'qwen-vl-ocr-latest',
              description: '自动指向最新版本',
            },
            {
              name: 'qwen-vl-ocr-2025-11-20',
              description: '固定日期版本',
            },
            {
              name: 'qwen-vl-ocr',
              description: '稳定基础版本',
            },
          ],
        },
      },
      {
        id: 'openai',
        name: 'OpenAI',
        capabilities: ['llm', 'embedding', 'vlm', 'asr', 'tts'],
        accent: '#10a37f',
      },
    ]);
    mockedApi.models.mockResolvedValue([]);

    render(
      <MemoryRouter>
        <ModelSettingsPage />
      </MemoryRouter>,
    );

    const ocrFilter = await screen.findByRole('button', { name: 'OCR' });
    expect(screen.getByRole('button', { name: 'ASR' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'TTS' })).toBeInTheDocument();

    fireEvent.click(ocrFilter);
    expect(
      screen.getByRole('button', { name: '添加 qwen3.5-ocr' }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: '添加 qwen-vl-ocr-latest' }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', {
        name: '添加 qwen-vl-ocr-2025-11-20',
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: '添加 qwen-vl-ocr' }),
    ).toBeInTheDocument();
    fireEvent.click(
      screen.getByRole('button', { name: '添加 qwen3.5-ocr' }),
    );

    expect(screen.getByRole('combobox', { name: '模型类型' })).toHaveValue(
      'ocr',
    );
    expect(screen.getByRole('textbox', { name: '模型名称' })).toHaveValue(
      'qwen3.5-ocr',
    );
    expect(screen.queryByText('推荐')).not.toBeInTheDocument();
  });

  it('shows official VLM and RERANK presets without recommendation labels', async () => {
    mockedApi.providers.mockResolvedValue([
      {
        id: 'tongyi-qianwen',
        name: 'Tongyi-Qianwen',
        capabilities: ['llm', 'embedding', 'rerank', 'vlm', 'ocr', 'asr', 'tts'],
        accent: '#615ced',
        model_presets: {
          rerank: [
            {
              name: 'qwen3-rerank',
              description: '适用于文本语义检索与 RAG 精排',
            },
          ],
          vlm: [
            {
              name: 'qwen3-vl-flash',
              description: 'Qwen3 系列低延迟视觉理解模型',
            },
            {
              name: 'qwen3-vl-plus',
              description: 'Qwen3 系列高能力视觉理解模型',
            },
          ],
        },
      },
      {
        id: 'openai',
        name: 'OpenAI',
        capabilities: ['llm', 'embedding', 'vlm', 'asr', 'tts'],
        accent: '#10a37f',
      },
    ]);
    mockedApi.models.mockResolvedValue([]);

    render(
      <MemoryRouter>
        <ModelSettingsPage />
      </MemoryRouter>,
    );

    fireEvent.click(await screen.findByRole('button', { name: 'VLM' }));
    expect(
      screen.getByRole('button', { name: '添加 qwen3-vl-flash' }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: '添加 qwen3-vl-plus' }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: '添加 OpenAI' }),
    ).toBeInTheDocument();
    expect(screen.getAllByText('Tongyi-Qianwen · VLM')).toHaveLength(2);
    expect(screen.queryByText('推荐')).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'RERANK' }));
    expect(
      screen.getByRole('button', { name: '添加 qwen3-rerank' }),
    ).toBeInTheDocument();
    expect(screen.getByText('Tongyi-Qianwen · RERANK')).toBeInTheDocument();
    expect(screen.queryByText('推荐')).not.toBeInTheDocument();

    fireEvent.click(
      screen.getByRole('button', { name: '添加 qwen3-rerank' }),
    );
    expect(screen.getByRole('combobox', { name: '模型类型' })).toHaveValue(
      'rerank',
    );
    expect(screen.getByRole('textbox', { name: '模型名称' })).toHaveValue(
      'qwen3-rerank',
    );
    expect(screen.getByRole('textbox', { name: 'API 地址' })).toHaveValue(
      'https://dashscope.aliyuncs.com/compatible-api/v1',
    );
  });
});
