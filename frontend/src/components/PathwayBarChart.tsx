import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell, LabelList } from "recharts";
import { Card } from "@/components/ui/card";

interface PathwayBarChartProps {
  data: Array<{
    id: string;
    name: string;
    source: string;
    NES?: number;
    FDR?: number;
    pval?: number;
    score?: number;
  }>;
  title: string;
  maxItems?: number;
}

export const PathwayBarChart = ({ data, title, maxItems = 20 }: PathwayBarChartProps) => {
  const normalizedData = data.map(item => {
    const baseScore = item.NES ?? item.score ?? (item.pval ? -Math.log10(item.pval) : 0);
    const value = Number.isFinite(baseScore) ? baseScore : 0;
    const displayName = item.name || item.id || "Pathway";
    return {
      ...item,
      value,
      displayName,
    };
  });

  const sortedData = normalizedData
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, maxItems)
    .reverse();

  const getColor = (item: any) => {
    if (item.value > 0) return "#10b981";
    if (item.value < 0) return "#ef4444";
    return "#94a3b8";
  };

  return (
    <Card className="p-6">
      <h3 className="text-lg font-semibold mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={Math.max(400, sortedData.length * 30)}>
        <BarChart
          layout="vertical"
          data={sortedData}
          margin={{ top: 5, right: 30, left: 200, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" />
          <YAxis 
            type="category" 
            dataKey="displayName" 
            width={220}
            tick={{ fontSize: 12 }}
          />
          <Tooltip
            content={({ active, payload }) => {
              if (active && payload && payload[0]) {
                const data = payload[0].payload;
                return (
                  <div className="bg-background border border-border rounded-lg p-3 shadow-lg">
                    <p className="font-semibold">{data.displayName || data.name}</p>
                    <p className="text-sm">Source: {data.source}</p>
                    {data.NES !== undefined && <p className="text-sm">NES: {data.NES.toFixed(3)}</p>}
                    {data.value !== undefined && data.NES === undefined && (
                      <p className="text-sm">Score: {data.value.toFixed(3)}</p>
                    )}
                    {data.pval !== undefined && <p className="text-sm">p-value: {data.pval.toFixed(4)}</p>}
                    {data.FDR !== undefined && <p className="text-sm">FDR: {data.FDR.toFixed(4)}</p>}
                  </div>
                );
              }
              return null;
            }}
          />
          <Bar dataKey="value" name="Score">
            {sortedData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getColor(entry)} />
            ))}
            <LabelList dataKey="value" position="right" formatter={(val: number) => val.toFixed(2)} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="flex items-center justify-center gap-4 mt-4 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-green-500 rounded" />
          <span>Enriched (positive NES)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-red-500 rounded" />
          <span>Depleted (negative NES)</span>
        </div>
      </div>
    </Card>
  );
};

