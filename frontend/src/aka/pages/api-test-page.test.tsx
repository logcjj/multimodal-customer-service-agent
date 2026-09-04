import { api } from '@/aka/api/client';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import ApiTestPage from './api-test-page';

jest.mock('@/aka/api/client', () => ({
  api: {
    chat: jest.fn(),
  },
}));

jest.mock('@/aka/components/message-content', () => ({
  MessageContent: ({ children }: { children: string }) => <div>{children}</div>,
}));

jest.mock('@/aka/lib/conversations', () => ({
  getBrowserClientId: jest.fn(() => 'anon-api-test'),
}));

const mockedApi = api as { chat: jest.Mock };

class TestFileReader {
  result: string | ArrayBuffer | null = null;
  onload: FileReader['onload'] = null;
  onerror: FileReader['onerror'] = null;

  readAsDataURL(file: Blob) {
    this.result = `data:${file.type};base64,dGVzdA==`;
    this.onload?.(new ProgressEvent('load') as ProgressEvent<FileReader>);
  }
}

const nativeFileReader = globalThis.FileReader;

describe('API test page', () => {
  beforeAll(() => {
    Object.defineProperty(globalThis, 'FileReader', {
      configurable: true,
      writable: true,
      value: TestFileReader,
    });
  });

  afterAll(() => {
    Object.defineProperty(globalThis, 'FileReader', {
      configurable: true,
      writable: true,
      value: nativeFileReader,
    });
  });

  beforeEach(() => {
    mockedApi.chat.mockReset();
    mockedApi.chat.mockResolvedValue({
      request_id: 'req-api-test',
      session_id: 'session-api-test',
      answer: '**滤网清洁说明**',
      route: 'technical_knowledge',
      citations: [
        {
          evidence_id: 'evidence-1',
          source_type: 'manual',
          title: '空气净化器使用说明书',
          child_ids: [],
          image_chunk_ids: [],
          asset_ids: [],
          score_breakdown: {},
        },
      ],
      assets: [],
      verification: {
        passed: true,
        action: 'accepted',
        confidence: 0.93,
        issues: [],
      },
      trace: {
        request_id: 'req-api-test',
        session_id: 'session-api-test',
        route: 'technical_knowledge',
        selected_agents: [],
        steps: [],
        spans: [],
        fallback_reason: null,
        total_latency_ms: 265,
      },
      used_legacy: false,
      timestamp: 1_752_000_000,
    });
  });

  it('sends a customer-service request and renders the real response metadata', async () => {
    render(<ApiTestPage />);

    fireEvent.click(screen.getByRole('button', { name: '发送测试' }));

    await waitFor(() =>
      expect(mockedApi.chat).toHaveBeenCalledWith({
        question: '空气净化器滤网如何清洁？',
        images: [],
        user_id: 'anon-api-test',
      }),
    );
    expect(await screen.findByText('technical_knowledge')).toBeInTheDocument();
    expect(screen.getByText('请求 req-api-test')).toBeInTheDocument();
    expect(screen.getByText('质量校验通过')).toBeInTheDocument();
    expect(screen.getByText('空气净化器使用说明书')).toBeInTheDocument();
  });

  it('includes a supplied session id in the request', async () => {
    render(<ApiTestPage />);

    fireEvent.change(screen.getByLabelText('Session ID'), {
      target: { value: 'existing-session' },
    });
    fireEvent.click(screen.getByRole('button', { name: '发送测试' }));

    await waitFor(() =>
      expect(mockedApi.chat).toHaveBeenCalledWith({
        question: '空气净化器滤网如何清洁？',
        images: [],
        session_id: 'existing-session',
        user_id: 'anon-api-test',
      }),
    );
  });

  it('encodes selected images and sends them with the request', async () => {
    render(<ApiTestPage />);

    const image = new File(['image-content'], 'panel.jpg', {
      type: 'image/jpeg',
    });
    fireEvent.change(screen.getByLabelText('选择图片文件'), {
      target: { files: [image] },
    });

    expect(await screen.findByAltText('panel.jpg')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '发送测试' }));

    await waitFor(() =>
      expect(mockedApi.chat).toHaveBeenCalledWith({
        question: '空气净化器滤网如何清洁？',
        images: ['data:image/jpeg;base64,dGVzdA=='],
        user_id: 'anon-api-test',
      }),
    );
  });
});
