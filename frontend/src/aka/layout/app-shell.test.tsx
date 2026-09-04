import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { AppShell } from './app-shell';

jest.mock('@/aka/api/client', () => ({
  api: {
    readiness: jest.fn().mockResolvedValue({ status: 'ready' }),
  },
}));

describe('AKA RAGFlow application shell', () => {
  it('renders the operational routes using the RAGFlow navigation language', async () => {
    render(
      <MemoryRouter initialEntries={['/']}>
        <AppShell>
          <div>页面内容</div>
        </AppShell>
      </MemoryRouter>,
    );

    expect(screen.getByRole('link', { name: '知识库' })).toHaveAttribute(
      'href',
      '/datasets',
    );
    expect(screen.getByRole('link', { name: '对话' })).toHaveAttribute(
      'href',
      '/chat',
    );
    expect(screen.getByRole('link', { name: '智能体' })).toHaveAttribute(
      'href',
      '/agents',
    );
    expect(await screen.findByText('后端在线')).toBeInTheDocument();
    expect(screen.getByText('页面内容')).toBeInTheDocument();
  });

  it('uses the landscape layout fallback when orientation locking is unavailable', async () => {
    const { container } = render(
      <MemoryRouter initialEntries={['/chat']}>
        <AppShell>
          <div>Landscape content</div>
        </AppShell>
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByRole('button', { name: '切换横屏显示' }));

    await waitFor(() =>
      expect(
        container.querySelector('[data-forced-landscape="true"]'),
      ).toBeInTheDocument(),
    );
    expect(
      screen.getByRole('button', { name: '退出横屏显示' }),
    ).toHaveAttribute('aria-pressed', 'true');
  });
});
