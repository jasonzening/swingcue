import { requireAdminPage } from '@/lib/auth';
import { AnnotateWorkbench } from './AnnotateWorkbench';

export const dynamic = 'force-dynamic';

export default async function Page({
  params,
}: { params: Promise<{ videoId: string }> }) {
  await requireAdminPage();
  const { videoId } = await params;
  return <AnnotateWorkbench videoId={videoId} />;
}
