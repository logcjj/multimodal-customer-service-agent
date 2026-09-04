import { apiUrl } from '@/lib/runtime-paths';
import { Fragment } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const picturePlaceholderPattern = /<\s*PIC\s*>/gi;

interface MessageContentProps {
  children: string;
  assets?: string[];
}

function assetUrl(assetId: string) {
  return apiUrl(`/api/assets/${encodeURIComponent(assetId)}`);
}

function AnswerAsset({ assetId, index }: { assetId: string; index: number }) {
  const url = assetUrl(assetId);
  return (
    <a
      className="block overflow-hidden rounded-md border border-border-button bg-bg-card"
      href={url}
      target="_blank"
      rel="noreferrer"
    >
      <img
        className="block max-h-80 w-full object-contain"
        src={url}
        alt={`说明书关联图片 ${index + 1}`}
        loading="lazy"
      />
    </a>
  );
}

function MarkdownFragment({ children }: { children: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        p: ({ children: content }) => (
          <p className="mb-3 last:mb-0">{content}</p>
        ),
        ul: ({ children: content }) => (
          <ul className="mb-3 list-disc space-y-1 pl-5">{content}</ul>
        ),
        ol: ({ children: content }) => (
          <ol className="mb-3 list-decimal space-y-1 pl-5">{content}</ol>
        ),
        code: ({ children: content }) => (
          <code className="rounded bg-bg-component px-1 py-0.5 font-mono text-[0.92em]">
            {content}
          </code>
        ),
        strong: ({ children: content }) => (
          <strong className="font-semibold text-text-primary">{content}</strong>
        ),
        img: ({ alt, src }) => (
          <img
            className="mb-3 max-h-80 max-w-full rounded-md border border-border-button object-contain"
            src={src}
            alt={alt ?? ''}
            loading="lazy"
          />
        ),
      }}
    >
      {children}
    </ReactMarkdown>
  );
}

export function MessageContent({ children, assets = [] }: MessageContentProps) {
  const normalizedText = String(children ?? '');
  const uniqueAssets = Array.from(
    new Set(assets.map((asset) => asset.trim()).filter(Boolean)),
  );
  const textParts = normalizedText.split(picturePlaceholderPattern);
  const inlineAssetCount = Math.min(textParts.length - 1, uniqueAssets.length);

  return (
    <div className="aka-markdown">
      {textParts.map((part, index) => (
        <Fragment key={`message-part-${index}`}>
          {part.trim() ? <MarkdownFragment>{part}</MarkdownFragment> : null}
          {index < inlineAssetCount ? (
            <div className="mb-3" data-testid="inline-answer-asset">
              <AnswerAsset assetId={uniqueAssets[index]} index={index} />
            </div>
          ) : null}
        </Fragment>
      ))}
    </div>
  );
}
