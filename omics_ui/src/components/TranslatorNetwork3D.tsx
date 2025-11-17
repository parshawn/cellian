import { useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Sphere, Line, Text } from "@react-three/drei";
import * as THREE from "three";

interface TranslatorNetwork3DProps {
  hoveredElement: string | null;
  setHoveredElement: (element: string | null) => void;
  onNodeClick?: (nodeId: string) => void;
  completedNodes?: string[];
  activePath?: string[]; // Path that should be highlighted based on agent position
}

interface Node3DProps {
  position: [number, number, number];
  color: string;
  label: string;
  nodeId: string;
  hoveredElement: string | null;
  setHoveredElement: (element: string | null) => void;
  onNodeClick?: (nodeId: string) => void;
  isActive: boolean;
  isHighlighted: boolean;
}

const Node3D = ({
  position,
  color,
  label,
  nodeId,
  hoveredElement,
  setHoveredElement,
  onNodeClick,
  isActive,
  isHighlighted,
}: Node3DProps) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const isHovered = hoveredElement === nodeId;

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.01;
    }
  });

  const handleClick = () => {
    onNodeClick?.(nodeId);
  };

  const handlePointerEnter = () => {
    setHoveredElement(nodeId);
  };

  const handlePointerLeave = () => {
    setHoveredElement(null);
  };

  const intensity = isHighlighted ? 1.5 : isActive ? 1.2 : isHovered ? 1.0 : 0.3;
  const scale = isHighlighted ? 1.3 : isActive ? 1.2 : isHovered ? 1.1 : 1.0;

  return (
    <group position={position}>
      <mesh
        ref={meshRef}
        onClick={handleClick}
        onPointerEnter={handlePointerEnter}
        onPointerLeave={handlePointerLeave}
        scale={scale}
      >
        <sphereGeometry args={[0.4, 32, 32]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={intensity}
          metalness={0.6}
          roughness={0.3}
        />
      </mesh>
      {/* Text Label - Bigger */}
      <Text
        position={[0, -0.9, 0]}
        fontSize={0.6}
        color="white"
        anchorX="center"
        anchorY="middle"
        outlineWidth={0.04}
        outlineColor="#000000"
      >
        {label}
      </Text>
    </group>
  );
};

interface Connection3DProps {
  start: [number, number, number];
  end: [number, number, number];
  color: string;
  isActive: boolean;
  label?: string;
}

const Connection3D = ({ start, end, color, isActive }: Connection3DProps) => {
  const points: [number, number, number][] = [start, end];
  
  // Calculate midpoint for arrow
  const midPoint = new THREE.Vector3(...start).add(new THREE.Vector3(...end)).multiplyScalar(0.5);

  return (
    <>
      <Line
        points={points}
        color={color}
        lineWidth={isActive ? 4 : 2}
        transparent
        opacity={isActive ? 1 : 0.4}
      />
      {/* Arrow head */}
      {isActive && (
        <mesh position={[midPoint.x, midPoint.y, midPoint.z]}>
          <coneGeometry args={[0.08, 0.25, 8]} />
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.8} />
        </mesh>
      )}
    </>
  );
};

const NetworkScene = ({
  hoveredElement,
  setHoveredElement,
  onNodeClick,
  completedNodes = [],
  activePath = [],
}: TranslatorNetwork3DProps) => {
  // Node positions in 3D space - spread out more for bigger graph
  const nodePositions = {
    drug: [-3.5, 2.5, 0] as [number, number, number],
    perturbation: [-3.5, -2.5, 0] as [number, number, number],
    rna: [0, 0, 0] as [number, number, number],
    protein: [3.5, 0, 0] as [number, number, number],
  };

  const nodeColors = {
    drug: "#3b82f6",
    perturbation: "#ef4444",
    rna: "#f59e0b",
    protein: "#ec4899",
  };

  // Determine which nodes and connections are active
  const isNodeActive = (nodeId: string) => {
    return completedNodes.includes(nodeId) || activePath.includes(nodeId);
  };

  const isConnectionActive = (connectionId: string) => {
    // Check if connection is in completed nodes or active path
    if (completedNodes.includes(connectionId)) return true;
    // Check if both start and end nodes are in active path
    const connectionMap: Record<string, [string, string]> = {
      "drug-to-rna": ["drug", "rna"],
      "perturb-to-rna": ["perturbation", "rna"],
      "rna-to-protein": ["rna", "protein"],
    };
    const [startNode, endNode] = connectionMap[connectionId] || ["", ""];
    return activePath.includes(startNode) && activePath.includes(endNode);
  };

  const isNodeHighlighted = (nodeId: string) => {
    // Highlight if it's the current active node in the path
    return activePath.length > 0 && activePath[activePath.length - 1] === nodeId;
  };

  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[5, 5, 5]} intensity={1} />
      <pointLight position={[-5, -5, -5]} intensity={0.5} />

      {/* Connections */}
      {/* Drug -> RNA */}
      <Connection3D
        start={nodePositions.drug}
        end={nodePositions.rna}
        color={isConnectionActive("drug-to-rna") ? "#10b981" : "#94a3b8"}
        isActive={isConnectionActive("drug-to-rna")}
      />
      {/* Gene Perturbation -> RNA */}
      <Connection3D
        start={nodePositions.perturbation}
        end={nodePositions.rna}
        color={isConnectionActive("perturb-to-rna") ? "#10b981" : "#94a3b8"}
        isActive={isConnectionActive("perturb-to-rna")}
      />
      {/* RNA -> Protein */}
      <Connection3D
        start={nodePositions.rna}
        end={nodePositions.protein}
        color={isConnectionActive("rna-to-protein") ? "#10b981" : "#94a3b8"}
        isActive={isConnectionActive("rna-to-protein")}
      />

      {/* Nodes */}
      <Node3D
        position={nodePositions.drug}
        color={nodeColors.drug}
        label="Drug Perturbation"
        nodeId="drug"
        hoveredElement={hoveredElement}
        setHoveredElement={setHoveredElement}
        onNodeClick={onNodeClick}
        isActive={isNodeActive("drug")}
        isHighlighted={isNodeHighlighted("drug")}
      />
      <Node3D
        position={nodePositions.perturbation}
        color={nodeColors.perturbation}
        label="Gene Perturbation"
        nodeId="perturbation"
        hoveredElement={hoveredElement}
        setHoveredElement={setHoveredElement}
        onNodeClick={onNodeClick}
        isActive={isNodeActive("perturbation")}
        isHighlighted={isNodeHighlighted("perturbation")}
      />
      <Node3D
        position={nodePositions.rna}
        color={nodeColors.rna}
        label="RNA"
        nodeId="rna"
        hoveredElement={hoveredElement}
        setHoveredElement={setHoveredElement}
        onNodeClick={onNodeClick}
        isActive={isNodeActive("rna")}
        isHighlighted={isNodeHighlighted("rna")}
      />
      <Node3D
        position={nodePositions.protein}
        color={nodeColors.protein}
        label="Protein"
        nodeId="protein"
        hoveredElement={hoveredElement}
        setHoveredElement={setHoveredElement}
        onNodeClick={onNodeClick}
        isActive={isNodeActive("protein")}
        isHighlighted={isNodeHighlighted("protein")}
      />

      <OrbitControls
        enablePan={false}
        enableZoom={true}
        minDistance={5}
        maxDistance={12}
        autoRotate={false}
        autoRotateSpeed={0.5}
      />
    </>
  );
};

export const TranslatorNetwork3D = (props: TranslatorNetwork3DProps) => {
  return (
    <div className="w-full h-[600px] rounded-lg overflow-hidden bg-gradient-to-br from-secondary/30 to-secondary/10">
      <Canvas camera={{ position: [0, 0, 8], fov: 50 }} style={{ width: '100%', height: '100%' }}>
        <NetworkScene {...props} />
      </Canvas>
    </div>
  );
};

