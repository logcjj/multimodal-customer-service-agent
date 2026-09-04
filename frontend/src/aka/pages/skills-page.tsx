import { api } from '@/aka/api/client';
import type { SkillDefinition, ToolDefinition } from '@/aka/api/types';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  CheckCircle2,
  Clock3,
  LockKeyhole,
  PackageOpen,
  Wrench,
} from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';

export default function SkillsPage() {
  const [skills, setSkills] = useState<SkillDefinition[]>([]);
  const [tools, setTools] = useState<ToolDefinition[]>([]);
  const [tab, setTab] = useState<'skills' | 'tools'>('skills');
  const [error, setError] = useState('');

  const load = useCallback(async () => {
    try {
      const [skillData, toolData] = await Promise.all([
        api.skills(),
        api.tools(),
      ]);
      setSkills(skillData);
      setTools(toolData);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '加载失败');
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="size-full overflow-auto px-5 py-6 lg:px-8 lg:py-8">
      <div className="mx-auto max-w-[1400px] space-y-5">
        <header className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h1 className="text-2xl font-semibold">Skills 与工具</h1>
            <p className="mt-1 text-sm text-text-secondary">
              {skills.length} Skills · {tools.length} Tools
            </p>
          </div>
          <div className="flex rounded-md border border-border-button bg-bg-card p-1">
            <Button
              variant={tab === 'skills' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setTab('skills')}
              aria-label="Skills"
            >
              <PackageOpen className="size-4" />
              Skills
            </Button>
            <Button
              variant={tab === 'tools' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setTab('tools')}
              aria-label="Tools"
            >
              <Wrench className="size-4" />
              Tools
            </Button>
          </div>
        </header>
        {error ? (
          <div className="rounded-md border border-state-error/40 p-3 text-sm text-state-error">
            {error}
          </div>
        ) : null}
        <div className="overflow-hidden rounded-md border border-border-button">
          {tab === 'skills' ? (
            <>
              <div className="hidden grid-cols-[minmax(240px,1fr)_180px_120px_120px] gap-4 border-b border-border-button bg-bg-card px-4 py-3 text-xs text-text-secondary md:grid">
                <span>名称</span>
                <span>Owner</span>
                <span>版本</span>
                <span>状态</span>
              </div>
              {skills.map((skill) => (
                <article
                  key={skill.id}
                  className="grid gap-3 border-b border-border-button px-4 py-4 last:border-b-0 md:grid-cols-[minmax(240px,1fr)_180px_120px_120px] md:items-center"
                >
                  <div>
                    <div className="text-sm font-medium">{skill.name}</div>
                    <div className="mt-1 text-xs text-text-secondary">
                      {skill.description}
                    </div>
                  </div>
                  <code className="text-xs">{skill.owner}</code>
                  <span className="text-xs">{skill.version}</span>
                  <div>
                    <Badge variant="success">
                      <CheckCircle2 className="mr-1 size-3" />
                      {skill.status}
                    </Badge>
                  </div>
                </article>
              ))}
            </>
          ) : (
            <>
              <div className="hidden grid-cols-[minmax(240px,1fr)_120px_140px_120px_100px] gap-4 border-b border-border-button bg-bg-card px-4 py-3 text-xs text-text-secondary md:grid">
                <span>工具</span>
                <span>风险</span>
                <span>超时</span>
                <span>确认</span>
                <span>幂等</span>
              </div>
              {tools.map((tool) => (
                <article
                  key={tool.id}
                  className="grid gap-3 border-b border-border-button px-4 py-4 last:border-b-0 md:grid-cols-[minmax(240px,1fr)_120px_140px_120px_100px] md:items-center"
                >
                  <div>
                    <div className="text-sm font-medium">{tool.name}</div>
                    <code className="mt-1 block text-[10px] text-text-secondary">
                      {tool.id}
                    </code>
                  </div>
                  <div>
                    <Badge variant="secondary">
                      <LockKeyhole className="mr-1 size-3" />
                      {tool.risk_level}
                    </Badge>
                  </div>
                  <span className="flex items-center gap-1 text-xs">
                    <Clock3 className="size-3.5" />
                    {tool.timeout_ms} ms
                  </span>
                  <span className="text-xs">
                    {tool.requires_confirmation ? '需要' : '无需'}
                  </span>
                  <span className="text-xs">
                    {tool.idempotent ? '是' : '否'}
                  </span>
                </article>
              ))}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
