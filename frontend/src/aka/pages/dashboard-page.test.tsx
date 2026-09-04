import { api } from '@/aka/api/client';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { CLIENT_ID_STORAGE_KEY } from '../lib/conversations';
import DashboardPage from './dashboard-page';

jest.mock('@/aka/api/client', () => ({
  api: {
    readiness: jest.fn(),
    capabilities: jest.fn(),
    datasets: jest.fn(),
    traces: jest.fn(),
  },
}));

const mockedApi = api as {
  readiness: jest.Mock;
  capabilities: jest.Mock;
  datasets: jest.Mock;
  traces: jest.Mock;
};

describe('RAGFlow-style operational dashboard', () => {
  beforeEach(() => {
    window.localStorage.clear();
    window.localStorage.setItem(CLIENT_ID_STORAGE_KEY, 'anon-dashboard-owner');
    mockedApi.readiness.mockResolvedValue({
      status: 'ready',
      rollout_mode: 'agent_first',
      legacy_available: true,
      model_registry: 'sqlite',
      trace_store: 'sqlite',
      llm_configured: true,
      llm_model: 'deepseek-v4-flash',
      vlm_configured: false,
      embedding_configured: false,
      rerank_configured: false,
      ocr_configured: false,
    });
    mockedApi.capabilities.mockResolvedValue({
      multi_agent: true,
      vision: true,
      skills: true,
      mcp: true,
      memory: true,
      trace: true,
      legacy_fallback: true,
      stream_mode: 'status-events',
      agent_count: 6,
      skill_count: 8,
      tool_count: 11,
    });
    mockedApi.datasets.mockResolvedValue([
      {
        id: 'v6-manuals',
        name: 'V6 图文说明书知识库',
        description: '39 类产品说明书',
        parser_profile: 'manual',
        visibility: 'private',
        published_version: 'v6-import-v1',
        status: 'ready',
        is_system: true,
        retrieval_profile_id: null,
        document_count: 39,
        parent_count: 1943,
        child_count: 6570,
        asset_count: 2563,
        failed_job_count: 0,
        created_at: '2026-07-23T00:00:00',
        updated_at: '2026-07-23T00:00:00',
      },
    ]);
    mockedApi.traces.mockResolvedValue([]);
  });

  it('shows actual model health, retrieval mode, knowledge metrics and capabilities', async () => {
    render(
      <MemoryRouter>
        <DashboardPage />
      </MemoryRouter>,
    );

    expect(await screen.findByText('deepseek-v4-flash')).toBeInTheDocument();
    expect(screen.getByText('lexical-only')).toBeInTheDocument();
    expect(screen.getByText('6,570')).toBeInTheDocument();
    expect(screen.getByText('1,943')).toBeInTheDocument();
    expect(screen.getByText('6 Agents')).toBeInTheDocument();
    expect(mockedApi.traces).toHaveBeenCalledWith('anon-dashboard-owner');
    expect(window.localStorage.getItem(CLIENT_ID_STORAGE_KEY)).toBe(
      'anon-dashboard-owner',
    );
  });
});
