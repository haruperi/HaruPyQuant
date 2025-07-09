import dynamic from "next/dynamic";
import type { PlotParams } from "react-plotly.js";

// This type assertion helps TypeScript recognize the props
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false }) as React.ComponentType<PlotParams>;

export default Plot; 