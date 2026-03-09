import type { Metadata, Viewport } from "next";
import { Inter, Plus_Jakarta_Sans } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const plusJakartaSans = Plus_Jakarta_Sans({
  variable: "--font-plus-jakarta",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700", "800"],
});

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
};

export const metadata: Metadata = {
  title: "TrackLive — Real-Time AI Object Detection & Tracking",
  description:
    "Portfolio-grade real-time object detection and tracking powered by YOLOv8 + SORT. Features live WebSocket streaming, trajectory trails, virtual tripwire, class filtering, and CSV export.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${plusJakartaSans.variable}`}>
      <body className="antialiased">
        <TooltipProvider>
          {children}
          <Toaster />
          <Sonner
            theme="dark"
            toastOptions={{
              style: {
                background: "hsl(220 25% 8%)",
                border: "1px solid hsl(187 100% 42% / 0.3)",
                color: "hsl(200 100% 95%)",
              },
            }}
          />
        </TooltipProvider>
      </body>
    </html>
  );
}
