"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export interface FpsDataPoint {
  t: number;
  fps: number;
  tracks: number;
}

interface FpsChartProps {
  data: FpsDataPoint[];
}

const FpsChart = ({ data }: FpsChartProps) => {
  return (
    <div className="glass-panel p-4 space-y-3 animate-fade-in">
      <div className="flex items-center justify-between">
        <h3 className="font-display text-xs font-bold tracking-wider text-foreground">
          PERFORMANCE CHART
        </h3>
        <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-sm bg-neon-cyan/70" />
            <span>FPS</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-sm bg-neon-magenta/70" />
            <span>TRACKS</span>
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={120}>
        <AreaChart data={data} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="fpsGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="hsl(187 100% 42%)" stopOpacity={0.4} />
              <stop offset="95%" stopColor="hsl(187 100% 42%)" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="tracksGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="hsl(292 84% 61%)" stopOpacity={0.4} />
              <stop offset="95%" stopColor="hsl(292 84% 61%)" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(220 20% 18% / 0.5)" />
          <XAxis dataKey="t" hide />
          <YAxis tick={{ fontSize: 9, fill: "hsl(200 20% 60%)" }} />
          <Tooltip
            contentStyle={{
              background: "hsl(220 25% 8%)",
              border: "1px solid hsl(187 100% 42% / 0.3)",
              borderRadius: "6px",
              fontSize: "11px",
              color: "hsl(200 100% 95%)",
            }}
            formatter={(value: number | string | undefined, name: string | undefined) => {
              if (typeof value === 'number') {
                return [value.toFixed(1), name === "fps" ? "FPS" : "Tracks"];
              }
              return [value || "0", name === "fps" ? "FPS" : "Tracks"];
            }}
            labelFormatter={() => ""}
          />
          <Area
            type="monotone"
            dataKey="fps"
            stroke="hsl(187 100% 42%)"
            strokeWidth={1.5}
            fill="url(#fpsGrad)"
            dot={false}
            isAnimationActive={false}
          />
          <Area
            type="monotone"
            dataKey="tracks"
            stroke="hsl(292 84% 61%)"
            strokeWidth={1.5}
            fill="url(#tracksGrad)"
            dot={false}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export default FpsChart;
