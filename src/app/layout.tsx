import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SwingCue — AI Golf Swing Coach",
  description:
    "Upload your golf swing. See what's wrong. Get one clear fix — no jargon, no guesswork. SwingCue is the AI coach in your pocket.",
  keywords: ["golf", "swing analysis", "AI golf coach", "golf improvement", "swing fix"],
  authors: [{ name: "SwingCue" }],
  openGraph: {
    title: "SwingCue — AI Golf Swing Coach",
    description: "Upload your swing. See what's wrong. Know what to fix next.",
    url: "https://swingcue.ai",
    siteName: "SwingCue",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "SwingCue — AI Golf Swing Coach",
    description: "Upload your swing. See what's wrong. Know what to fix next.",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#080c08",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
