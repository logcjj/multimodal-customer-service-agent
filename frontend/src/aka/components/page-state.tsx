import { Button } from '@/components/ui/button';
import { AlertCircle, Inbox, RefreshCw } from 'lucide-react';

export function PageLoading({ label = '正在加载' }: { label?: string }) {
  const skeleton = 'animate-pulse rounded-md bg-bg-card';
  return (
    <div className="space-y-3" aria-label={label} role="status">
      <div className={`${skeleton} h-8 w-48`} />
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        {Array.from({ length: 4 }, (_, index) => (
          <div key={index} className={`${skeleton} h-32`} />
        ))}
      </div>
    </div>
  );
}

export function PageError({
  message,
  onRetry,
}: {
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div className="grid min-h-72 place-items-center text-center">
      <div className="max-w-md space-y-3">
        <AlertCircle className="mx-auto size-8 text-state-error" />
        <div className="font-medium">数据加载失败</div>
        <p className="text-sm text-text-secondary">{message}</p>
        {onRetry && (
          <Button variant="outline" onClick={onRetry}>
            <RefreshCw className="size-4" />
            重新加载
          </Button>
        )}
      </div>
    </div>
  );
}

export function PageEmpty({
  title,
  description,
}: {
  title: string;
  description: string;
}) {
  return (
    <div className="grid min-h-72 place-items-center text-center">
      <div className="max-w-sm space-y-2">
        <Inbox className="mx-auto size-8 text-text-secondary" />
        <div className="font-medium">{title}</div>
        <p className="text-sm text-text-secondary">{description}</p>
      </div>
    </div>
  );
}
