'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import type { VideoListEntry } from '@/lib/types/annotation';

/**
 * Admin landing page for the landmark annotation workbench (PR-7A).
 *
 * Fetches GET /api/admin/videos and renders the list — WHAM-ready
 * videos first (full opacity, prominent badge), pre-WHAM videos
 * de-emphasized (opacity 0.5) but still clickable so we can spot
 * data gaps. Click row → /admin/annotate/[videoId].
 *
 * Styling: matches existing app's single-file `const css = ...` pattern.
 * Pure black/gray/white minimalist palette; --annot-error reserved for
 * the error line only.
 */
export function AnnotateHome() {
  const router = useRouter();
  const [videos, setVideos] = useState<VideoListEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch('/api/admin/videos', { cache: 'no-store' });
        if (!res.ok) {
          if (!cancelled) setError(`Load failed: HTTP ${res.status}`);
          return;
        }
        const body = (await res.json()) as { videos: VideoListEntry[] };
        if (!cancelled) setVideos(body.videos);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'unknown error');
      }
    })();
    return () => { cancelled = true; };
  }, []);

  return (
    <div className="ah-root">
      <header className="ah-header">
        <div className="ah-title">Landmark Annotation Workbench</div>
        <div className="ah-subtitle">Internal · PR-7A</div>
      </header>

      <main className="ah-main">
        {error && <div className="ah-error">{error}</div>}

        {!error && videos === null && (
          <div className="ah-muted">Loading…</div>
        )}

        {!error && videos !== null && videos.length === 0 && (
          <div className="ah-muted">No videos found</div>
        )}

        {!error && videos !== null && videos.length > 0 && (
          <ul className="ah-list">
            {videos.map(v => {
              const filename = v.original_filename ?? `(${v.id.slice(0, 8)})`;
              const created = new Date(v.created_at).toLocaleDateString('en-US', {
                month: 'short', day: 'numeric',
              });
              return (
                <li
                  key={v.id}
                  className={`ah-row ${v.hasWham ? '' : 'ah-row-no-wham'}`}
                  onClick={() => router.push(`/admin/annotate/${v.id}`)}
                >
                  <div className="ah-row-main">
                    <div className="ah-row-name">{filename}</div>
                    <div className="ah-row-meta">
                      <span>{v.view_type === 'face_on' ? 'face-on' : 'down-the-line'}</span>
                      <span>·</span>
                      <span>{created}</span>
                      <span>·</span>
                      <span className="ah-row-id">{v.id.slice(0, 8)}</span>
                    </div>
                  </div>
                  <div className="ah-row-side">
                    <span className={`ah-pill ${v.hasWham ? 'ah-pill-on' : 'ah-pill-off'}`}>
                      {v.hasWham
                        ? `WHAM ${v.whamMeta?.frame_count ?? 0} frames`
                        : 'no WHAM'}
                    </span>
                    <span className="ah-ann-count">
                      {v.annotationCount} annotated
                    </span>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </main>
      <style>{css}</style>
    </div>
  );
}

const css = `
  .ah-root {
    min-height: 100vh;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-family: 'DM Sans', system-ui, sans-serif;
  }
  .ah-header {
    padding: 16px 24px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    display: flex;
    align-items: baseline;
    gap: 12px;
  }
  .ah-title {
    font-size: 16px;
    font-weight: 700;
    letter-spacing: -0.01em;
  }
  .ah-subtitle {
    font-size: 12px;
    color: var(--text-muted);
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  .ah-main {
    max-width: 880px;
    margin: 0 auto;
    padding: 24px 16px 48px;
  }
  .ah-muted {
    color: var(--text-muted);
    font-size: 14px;
    padding: 24px;
    text-align: center;
  }
  .ah-error {
    color: var(--annot-error);
    font-size: 13px;
    padding: 12px 14px;
    border: 1px solid var(--annot-error);
    border-radius: 4px;
    margin-bottom: 16px;
  }
  .ah-list {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .ah-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    padding: 14px 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    cursor: pointer;
    background: var(--surface-card);
    transition: background 0.12s, border-color 0.12s;
  }
  .ah-row:hover {
    background: rgba(255, 255, 255, 0.04);
    border-color: rgba(255, 255, 255, 0.18);
  }
  .ah-row-no-wham {
    opacity: 0.5;
  }
  .ah-row-main {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .ah-row-name {
    font-size: 14px;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .ah-row-meta {
    font-size: 11px;
    color: var(--text-muted);
    display: flex;
    gap: 6px;
    align-items: center;
  }
  .ah-row-id {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  .ah-row-side {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 4px;
    flex-shrink: 0;
  }
  .ah-pill {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 3px 9px;
    border-radius: 100px;
    border: 1px solid rgba(255, 255, 255, 0.2);
  }
  .ah-pill-on {
    border-color: var(--text-primary);
    color: var(--text-primary);
  }
  .ah-pill-off {
    color: var(--text-muted);
    border-color: rgba(255, 255, 255, 0.12);
  }
  .ah-ann-count {
    font-size: 11px;
    color: var(--text-muted);
  }
`;
