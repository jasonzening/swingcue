# NEXTJS_APP_ROUTER_STRUCTURE

## Planned structure
- app/(marketing)/page.tsx
- app/(auth)/sign-in/page.tsx
- app/(auth)/auth/callback/route.ts
- app/(protected)/layout.tsx
- app/(protected)/upload/page.tsx
- app/(protected)/processing/[videoId]/page.tsx
- app/(protected)/result/[videoId]/page.tsx
- app/(protected)/history/page.tsx
- app/(protected)/account/page.tsx

## API routes
- app/api/videos/upload/route.ts
- app/api/videos/[id]/analyze/route.ts
- app/api/videos/[id]/status/route.ts
- app/api/videos/[id]/result/route.ts
- app/api/history/route.ts
