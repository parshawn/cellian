import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Check } from "lucide-react";

type Condition = "Control" | "IFNγ" | "Co-culture";

interface ConditionSelectorProps {
  onConditionSelect: (condition: Condition) => void;
}

export const ConditionSelector = ({ onConditionSelect }: ConditionSelectorProps) => {
  const [selected, setSelected] = useState<Condition | null>(null);

  const handleSelect = (condition: Condition) => {
    setSelected(condition);
    onConditionSelect(condition);
  };

  const conditions: Condition[] = ["Control", "IFNγ", "Co-culture"];

  return (
    <div className="bg-card/80 backdrop-blur rounded-xl border border-border p-6 shadow-lg">
      <h3 className="text-sm font-semibold text-foreground mb-4">Select Condition</h3>
      <div className="space-y-2">
        {conditions.map((condition) => (
          <Button
            key={condition}
            variant={selected === condition ? "default" : "outline"}
            className={`w-full justify-start ${
              selected === condition
                ? "bg-gradient-to-r from-dna to-protein text-white"
                : ""
            }`}
            onClick={() => handleSelect(condition)}
          >
            {selected === condition && <Check className="w-4 h-4 mr-2" />}
            {condition}
          </Button>
        ))}
      </div>
      {selected && (
        <p className="text-xs text-muted-foreground mt-4">
          Selected: <span className="font-semibold">{selected}</span>
        </p>
      )}
    </div>
  );
};

