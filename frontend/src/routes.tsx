import { AppShell } from '@/aka/layout/app-shell';
import AgentsPage from '@/aka/pages/agents-page';
import ApiTestPage from '@/aka/pages/api-test-page';
import ChatPage from '@/aka/pages/chat-page';
import DashboardPage from '@/aka/pages/dashboard-page';
import DatasetDetailPage from '@/aka/pages/dataset-detail-page';
import DatasetsPage from '@/aka/pages/datasets-page';
import FilesPage from '@/aka/pages/files-page';
import McpPage from '@/aka/pages/mcp-page';
import ModelSettingsPage from '@/aka/pages/model-settings-page';
import SkillsPage from '@/aka/pages/skills-page';
import TracesPage from '@/aka/pages/traces-page';
import { Navigate, Outlet, createBrowserRouter } from 'react-router';

function ApplicationLayout() {
  return (
    <AppShell>
      <Outlet />
    </AppShell>
  );
}

export const routers = createBrowserRouter([
  {
    path: '/',
    element: <ApplicationLayout />,
    children: [
      { index: true, element: <DashboardPage /> },
      { path: 'datasets', element: <DatasetsPage /> },
      { path: 'dataset/:datasetId', element: <DatasetDetailPage /> },
      { path: 'chat', element: <ChatPage /> },
      { path: 'api-test', element: <ApiTestPage /> },
      { path: 'agents', element: <AgentsPage /> },
      { path: 'files', element: <FilesPage /> },
      { path: 'skills', element: <SkillsPage /> },
      { path: 'mcp', element: <McpPage /> },
      { path: 'traces', element: <TracesPage /> },
      { path: 'settings/models', element: <ModelSettingsPage /> },
      { path: '*', element: <Navigate to="/" replace /> },
    ],
  },
]);
