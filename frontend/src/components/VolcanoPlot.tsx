import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { Card } from "@/components/ui/card";

interface VolcanoPlotProps {
  data: Array<{
    gene: string;
    log2fc: number;
    pval: number;
    padj: number;
    direction: "up" | "down";
    significant: boolean;
  }>;
  title: string;
  xLabel?: string;
  yLabel?: string;
}

export const VolcanoPlot = ({ data, title, xLabel = "log₂ Fold Change", yLabel = "-log₁₀(p-value)" }: VolcanoPlotProps) => {
  const getColor = (point: any) => {
    if (!point.significant) return "#94a3b8"; // gray for non-significant
    return point.direction === "up" ? "#ef4444" : "#3b82f6"; // red for up, blue for down
  };

  const labeledPoints = [...data]
    .filter((point) => point.significant)
    .sort((a, b) => (b.y ?? 0) - (a.y ?? 0))
    .slice(0, 8);

  const renderLabel = (props: any) => {
    const { cx, cy, payload } = props;
    if (cx == null || cy == null) return null;
    return (
      <text
        x={cx + 6}
        y={cy - 6}
        fill="#475569"
        fontSize={10}
        fontWeight={600}
      >
        {payload.gene}
      </text>
    );
  };

  return (
    <Card className="p-6">
      <h3 className="text-lg font-semibold mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={400}>
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            type="number" 
            dataKey="x" 
            name={xLabel}
            label={{ value: xLabel, position: "insideBottom", offset: -5 }}
          />
          <YAxis 
            type="number" 
            dataKey="y" 
            name={yLabel}
            label={{ value: yLabel, angle: -90, position: "insideLeft" }}
          />
          <Tooltip 
            cursor={{ strokeDasharray: '3 3' }}
            content={({ active, payload }) => {
              if (active && payload && payload[0]) {
                const data = payload[0].payload;
                return (
                  <div className="bg-background border border-border rounded-lg p-3 shadow-lg">
                    <p className="font-semibold">{data.gene}</p>
                    <p className="text-sm">log₂FC: {data.log2fc.toFixed(3)}</p>
                    <p className="text-sm">p-value: {data.pval.toFixed(4)}</p>
                    <p className="text-sm">p-adj: {data.padj.toFixed(4)}</p>
                    <p className="text-sm">
                      <span className={data.direction === "up" ? "text-red-600" : "text-blue-600"}>
                        {data.direction === "up" ? "↑ Upregulated" : "↓ Downregulated"}
                      </span>
                    </p>
                  </div>
                );
              }
              return null;
            }}
          />
          <Scatter name="Genes" data={data} fill="#8884d8">
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getColor(entry)} />
            ))}
          </Scatter>
          {labeledPoints.length > 0 && (
            <Scatter
              data={labeledPoints}
              shape={renderLabel}
            />
          )}
        </ScatterChart>
      </ResponsiveContainer>
      <div className="flex items-center justify-center gap-4 mt-4 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-red-500 rounded-full" />
          <span>Upregulated (p-value &lt; 0.05)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-blue-500 rounded-full" />
          <span>Downregulated (p-value &lt; 0.05)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-gray-400 rounded-full" />
          <span>Not significant</span>
        </div>
      </div>
    </Card>
  );
};

