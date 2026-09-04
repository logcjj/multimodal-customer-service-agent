import { ThemeProvider } from '@/components/theme-provider';
import { Toaster as Sonner } from '@/components/ui/sonner';
import { TooltipProvider } from '@/components/ui/tooltip';
import { ThemeEnum } from '@/constants/common';
import { queryClient } from '@/query-client';
import { QueryClientProvider } from '@tanstack/react-query';
import { RouterProvider } from 'react-router';
import { routers } from './routes';

export default function App() {
  return (
    <TooltipProvider>
      <QueryClientProvider client={queryClient}>
        <ThemeProvider
          defaultTheme={ThemeEnum.Dark}
          storageKey="aka-ragflow-ui-theme"
        >
          <RouterProvider router={routers} />
          <Sonner position="top-right" expand richColors closeButton />
        </ThemeProvider>
      </QueryClientProvider>
    </TooltipProvider>
  );
}
