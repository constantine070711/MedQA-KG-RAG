import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MedQA v2 · 可溯源医疗知识工作台",
  description:
    "融合知识图谱、混合检索与可靠性决策的中文医疗知识辅助问答系统。",
  applicationName: "MedQA v2",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#f3f7f6",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
