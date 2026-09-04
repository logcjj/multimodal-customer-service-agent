import { api } from '@/aka/api/client';
import { useTheme } from '@/components/theme-provider';
import { Button } from '@/components/ui/button';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetTitle,
  SheetTrigger,
} from '@/components/ui/sheet';
import { ThemeEnum } from '@/constants/common';
import { cn } from '@/lib/utils';
import {
  Bot,
  Cable,
  Database,
  FileStack,
  House,
  Menu,
  MessageSquareText,
  Moon,
  RotateCw,
  Settings,
  Sun,
} from 'lucide-react';
import { useEffect, useState } from 'react';
import { Link, useLocation } from 'react-router';

const navigation = [
  { href: '/', label: '工作台', icon: House },
  { href: '/datasets', label: '知识库', icon: Database },
  { href: '/chat', label: '对话', icon: MessageSquareText },
  { href: '/api-test', label: '接口测试', icon: Cable },
  { href: '/agents', label: '智能体', icon: Bot },
  { href: '/files', label: '文件', icon: FileStack },
] as const;

function isActivePath(pathname: string, href: string) {
  if (href === '/') return pathname === '/';
  if (href === '/datasets' && pathname.startsWith('/dataset/')) return true;
  return pathname === href || pathname.startsWith(`${href}/`);
}

function NavigationLinks({ mobile = false }: { mobile?: boolean }) {
  const { pathname } = useLocation();

  return (
    <nav aria-label="主导航">
      <ul
        className={cn(
          mobile
            ? 'space-y-1'
            : 'flex items-center rounded-full border border-border-button bg-bg-card p-1',
        )}
      >
        {navigation.map(({ href, label, icon: Icon }) => {
          const active = isActivePath(pathname, href);
          return (
            <li key={href}>
              <Link
                to={href}
                aria-current={active ? 'page' : undefined}
                aria-label={label}
                className={cn(
                  'inline-flex items-center text-sm transition-colors',
                  mobile
                    ? 'w-full gap-3 rounded-md px-4 py-3 text-text-secondary hover:bg-bg-card hover:text-text-primary'
                    : 'h-9 justify-center gap-2 rounded-full px-4 text-text-secondary hover:text-text-primary',
                  active &&
                    (mobile
                      ? 'bg-bg-card font-medium text-text-primary'
                      : 'bg-text-primary text-bg-base shadow-sm hover:text-bg-base'),
                )}
              >
                <Icon className="size-4 stroke-[1.7]" />
                <span>{label}</span>
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

function ThemeToggle({ mobile = false }: { mobile?: boolean }) {
  const { theme, setTheme } = useTheme();
  const dark = theme === ThemeEnum.Dark;

  return (
    <Button
      variant="ghost"
      size="icon"
      className={cn(!mobile && 'hidden sm:inline-flex')}
      aria-label={dark ? '切换为浅色主题' : '切换为深色主题'}
      onClick={() => setTheme(dark ? ThemeEnum.Light : ThemeEnum.Dark)}
    >
      {dark ? <Moon className="size-4" /> : <Sun className="size-4" />}
    </Button>
  );
}

type LockableOrientation = ScreenOrientation & {
  lock?: (orientation: 'landscape') => Promise<void>;
  unlock?: () => void;
};

function useLandscapeMode() {
  const [forcedLandscape, setForcedLandscape] = useState(false);
  const [nativeLandscape, setNativeLandscape] = useState(false);

  useEffect(() => {
    const handleFullscreenChange = () => {
      if (!document.fullscreenElement) {
        setForcedLandscape(false);
        setNativeLandscape(false);
      }
    };
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, []);

  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return undefined;
    const landscapeQuery = window.matchMedia('(orientation: landscape)');
    const handleOrientationChange = () => {
      if (landscapeQuery.matches) setForcedLandscape(false);
    };
    landscapeQuery.addEventListener?.('change', handleOrientationChange);
    return () => {
      landscapeQuery.removeEventListener?.('change', handleOrientationChange);
    };
  }, []);

  async function toggleLandscape() {
    const orientation = window.screen.orientation as LockableOrientation;
    if (forcedLandscape || nativeLandscape) {
      orientation?.unlock?.();
      setForcedLandscape(false);
      setNativeLandscape(false);
      if (document.fullscreenElement && document.exitFullscreen) {
        await document.exitFullscreen().catch(() => undefined);
      }
      return;
    }

    try {
      if (
        !document.fullscreenElement &&
        document.documentElement.requestFullscreen
      ) {
        await document.documentElement.requestFullscreen({
          navigationUI: 'hide',
        });
      }
      if (orientation?.lock) {
        await orientation.lock('landscape');
        setNativeLandscape(true);
        return;
      }
    } catch {
      // iOS and embedded browsers commonly reject orientation locking.
    }
    setForcedLandscape(true);
  }

  return {
    active: forcedLandscape || nativeLandscape,
    forcedLandscape,
    toggleLandscape,
  };
}

function BackendStatus() {
  const [status, setStatus] = useState<'checking' | 'online' | 'offline'>(
    'checking',
  );

  useEffect(() => {
    let active = true;
    void api
      .readiness()
      .then(() => {
        if (active) setStatus('online');
      })
      .catch(() => {
        if (active) setStatus('offline');
      });
    return () => {
      active = false;
    };
  }, []);

  const label =
    status === 'online'
      ? '后端在线'
      : status === 'offline'
        ? '后端离线'
        : '检测后端';

  return (
    <div className="hidden items-center gap-2 rounded-full border border-border-button bg-bg-card px-3 py-1.5 text-xs text-text-secondary sm:flex">
      <span
        className={cn(
          'size-2 rounded-full',
          status === 'online'
            ? 'bg-state-success'
            : status === 'offline'
              ? 'bg-state-error'
              : 'bg-text-secondary',
        )}
      />
      {label}
    </div>
  );
}

export function AppShell({
  children,
}: {
  children?: import('react').ReactNode;
}) {
  const { active, forcedLandscape, toggleLandscape } = useLandscapeMode();

  return (
    <div
      className={cn(
        'aka-app-shell grid size-full min-w-0 grid-rows-[64px_1fr] bg-bg-base text-text-primary',
        forcedLandscape && 'aka-app-shell--forced-landscape',
      )}
      data-forced-landscape={forcedLandscape ? 'true' : 'false'}
    >
      <header className="aka-app-header flex min-w-0 items-center gap-3 border-b border-border-button px-4 lg:px-6">
        <div className="flex min-w-0 items-center gap-2 lg:w-64">
          <div className="lg:hidden">
            <Sheet>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon" aria-label="打开导航">
                  <Menu className="size-5" />
                </Button>
              </SheetTrigger>
              <SheetContent
                side="left"
                className="w-[min(86vw,19rem)] max-w-none rounded-none p-0"
              >
                <SheetTitle className="sr-only">主导航</SheetTitle>
                <SheetDescription className="sr-only">
                  打开 AKA Sentinel 的主要工作区
                </SheetDescription>
                <div className="flex h-16 items-center gap-3 border-b border-border-button px-5">
                  <img src="/logo.svg" alt="AKA Sentinel" className="size-8" />
                  <div>
                    <div className="text-sm font-semibold">AKA Sentinel</div>
                    <div className="text-xs text-text-secondary">
                      Multi-Agent Service OS
                    </div>
                  </div>
                  <div className="ml-auto">
                    <ThemeToggle mobile />
                  </div>
                </div>
                <div className="p-3">
                  <NavigationLinks mobile />
                  <Link
                    to="/settings/models"
                    className="mt-2 inline-flex w-full items-center gap-3 rounded-md border-t border-border-button px-4 py-3 text-sm text-text-secondary hover:bg-bg-card hover:text-text-primary"
                  >
                    <Settings className="size-4" />
                    <span>系统设置</span>
                  </Link>
                </div>
              </SheetContent>
            </Sheet>
          </div>

          <Link
            to="/"
            className="flex min-w-0 items-center gap-3"
            aria-label="AKA Sentinel 工作台"
          >
            <img src="/logo.svg" alt="" className="size-9 shrink-0" />
            <div className="hidden min-w-0 sm:block">
              <div className="truncate text-sm font-semibold tracking-[0]">
                AKA Sentinel
              </div>
              <div className="truncate text-[11px] text-text-secondary">
                Multi-Agent Service OS
              </div>
            </div>
          </Link>
        </div>

        <div className="hidden min-w-0 flex-1 justify-center lg:flex">
          <NavigationLinks />
        </div>

        <div className="ml-auto flex shrink-0 items-center gap-1 lg:w-64 lg:justify-end">
          <BackendStatus />
          <Button
            variant="ghost"
            size="icon"
            className="shrink-0 lg:hidden"
            aria-label={active ? '退出横屏显示' : '切换横屏显示'}
            aria-pressed={active}
            title={active ? '退出横屏显示' : '切换横屏显示'}
            onClick={() => void toggleLandscape()}
          >
            <RotateCw
              className={cn('size-4', active && 'text-accent-primary')}
            />
          </Button>
          <ThemeToggle />
          <Button
            asLink
            variant="ghost"
            size="icon"
            className="hidden sm:inline-flex"
            to="/settings/models"
            aria-label="系统设置"
          >
            <Settings className="size-4" />
          </Button>
          <div className="ml-1 hidden size-8 place-items-center rounded-full bg-accent-primary text-xs font-semibold text-white sm:grid">
            AK
          </div>
        </div>
      </header>

      <main className="aka-app-main min-h-0 min-w-0 overflow-hidden">
        {children}
      </main>
    </div>
  );
}
