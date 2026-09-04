import { QueryClient } from '@tanstack/react-query';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

export const vectorMapQueryKey = (
  datasetId: string,
  publishedVersion?: string | null,
) => ['aka', 'vector-map', datasetId, publishedVersion ?? 'unpublished'] as const;
