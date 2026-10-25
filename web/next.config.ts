import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  devIndicators: false,
  agentRules: false,
  output: "standalone",
};

export default nextConfig;
