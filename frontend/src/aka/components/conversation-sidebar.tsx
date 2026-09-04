import type { ConversationSummary } from '@/aka/api/types';
import { groupConversations } from '@/aka/lib/conversations';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Input } from '@/components/ui/input';
import { cn } from '@/lib/utils';
import { MessageSquare, MoreHorizontal, Pencil, Plus, Trash2, X } from 'lucide-react';
import { useState } from 'react';

interface ConversationSidebarProps {
  items: ConversationSummary[];
  activeId?: string;
  disabled: boolean;
  onNew: () => void;
  onSelect: (conversationId: string) => void;
  onRename: (conversationId: string, title: string) => void | Promise<void>;
  onDelete: (conversationId: string) => void | Promise<void>;
  onClose?: () => void;
}

export function ConversationSidebar({
  items,
  activeId,
  disabled,
  onNew,
  onSelect,
  onRename,
  onDelete,
  onClose,
}: ConversationSidebarProps) {
  const [editingId, setEditingId] = useState<string>();
  const [editingTitle, setEditingTitle] = useState('');
  const [pendingDeleteId, setPendingDeleteId] = useState<string>();
  const groups = groupConversations(items);

  function beginRename(item: ConversationSummary) {
    setEditingId(item.id);
    setEditingTitle(item.title);
  }

  function commitRename() {
    const title = editingTitle.trim();
    if (editingId && title) void onRename(editingId, title);
    setEditingId(undefined);
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-bg-card" aria-label="对话历史">
      <div className="flex h-14 shrink-0 items-center justify-between border-b border-border-button px-3">
        <div className="flex min-w-0 items-center gap-2 text-sm font-semibold">
          <MessageSquare className="size-4 shrink-0" />
          <span className="truncate">对话记录</span>
        </div>
        <div className="flex items-center gap-1">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label="新建对话"
            disabled={disabled}
            onClick={onNew}
          >
            <Plus className="size-4" />
          </Button>
          {onClose ? (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              aria-label="关闭对话历史"
              onClick={onClose}
            >
              <X className="size-4" />
            </Button>
          ) : null}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-2 py-3">
        {groups.length === 0 ? (
          <div className="px-2 py-8 text-center text-xs text-text-secondary">
            暂无历史对话
          </div>
        ) : (
          groups.map((group) => (
            <section key={group.key} className="mb-4 last:mb-0">
              <h2 className="px-2 pb-1.5 text-[11px] font-medium text-text-secondary">
                {group.label}
              </h2>
              <div className="space-y-0.5">
                {group.items.map((item) => {
                  const active = item.id === activeId;
                  const editing = item.id === editingId;
                  return (
                    <div
                      key={item.id}
                      className={cn(
                        'group relative flex min-h-9 items-center border-l-2',
                        active
                          ? 'border-accent-primary bg-accent-primary/8'
                          : 'border-transparent hover:bg-bg-component',
                      )}
                    >
                      {editing ? (
                        <Input
                          autoFocus
                          aria-label="对话标题"
                          className="mx-1 h-8 min-w-0"
                          value={editingTitle}
                          onChange={(event) => setEditingTitle(event.target.value)}
                          onBlur={commitRename}
                          onKeyDown={(event) => {
                            if (event.key === 'Enter') commitRename();
                            if (event.key === 'Escape') setEditingId(undefined);
                          }}
                        />
                      ) : (
                        <Button
                          type="button"
                          variant="ghost"
                          size="auto"
                          className="h-9 min-w-0 flex-1 justify-start rounded-none px-2 text-left text-xs font-normal"
                          aria-label={item.title}
                          aria-current={active || undefined}
                          disabled={disabled}
                          onClick={() => onSelect(item.id)}
                        >
                          <span className="truncate">{item.title}</span>
                        </Button>
                      )}
                      {!editing ? (
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button
                              type="button"
                              variant="ghost"
                              size="icon"
                              className="mr-0.5 size-8 shrink-0 opacity-100 xl:opacity-0 xl:group-hover:opacity-100 xl:focus-visible:opacity-100 data-[state=open]:opacity-100"
                              aria-label={`更多操作：${item.title}`}
                              disabled={disabled}
                            >
                              <MoreHorizontal className="size-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem onSelect={() => beginRename(item)}>
                              <Pencil className="size-4" />
                              重命名
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              className="text-state-error"
                              onSelect={() => setPendingDeleteId(item.id)}
                            >
                              <Trash2 className="size-4" />
                              删除
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      ) : null}
                    </div>
                  );
                })}
              </div>
            </section>
          ))
        )}
      </div>

      <AlertDialog
        open={Boolean(pendingDeleteId)}
        onOpenChange={(open) => {
          if (!open) setPendingDeleteId(undefined);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>删除这段对话？</AlertDialogTitle>
            <AlertDialogDescription>
              对话消息和当前会话记忆将一并删除，且无法恢复。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction
              aria-label="确认删除"
              className="bg-state-error text-white hover:bg-state-error/90"
              onClick={() => {
                if (pendingDeleteId) void onDelete(pendingDeleteId);
                setPendingDeleteId(undefined);
              }}
            >
              确认删除
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
