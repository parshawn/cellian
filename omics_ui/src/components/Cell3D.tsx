import { useRef, useState, useEffect } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Sphere, Float, MeshDistortMaterial, Line } from "@react-three/drei";
import * as THREE from "three";

interface Organelle3DProps {
  hoveredElement: string | null;
  setHoveredElement: (element: string | null) => void;
}

interface InjectionProps {
  isInjecting?: boolean;
  injectionProgress?: number;
  keepVisible?: boolean; // Keep visible after injection
}

interface Cell3DProps {
  hoveredElement: string | null;
  setHoveredElement: (element: string | null) => void;
  isInjecting: boolean;
  onInjectionComplete: () => void;
  showRNAChanges?: boolean;
  showProteinChanges?: boolean;
  perturbationType?: "gene" | "drug";
  selectedNode?: string | null;
  hasBothPerturbations?: boolean;
  hasGeneInjected?: boolean;
  hasDrugInjected?: boolean;
}

// Nucleus Component
const Nucleus = ({ hoveredElement, setHoveredElement }: Organelle3DProps) => {
  const nucleusRef = useRef<THREE.Mesh>(null);
  const isHovered = hoveredElement === "nucleus";

  useFrame((state) => {
    if (nucleusRef.current) {
      nucleusRef.current.rotation.y += 0.001;
    }
  });

  return (
    <Float speed={1.5} rotationIntensity={0.2} floatIntensity={0.3}>
      <Sphere
        ref={nucleusRef}
        args={[1.2, 32, 32]}
        position={[0, 0, 0]}
        onPointerEnter={() => setHoveredElement("nucleus")}
        onPointerLeave={() => setHoveredElement(null)}
      >
        <MeshDistortMaterial
          color={isHovered ? "#4299e1" : "#5da8e8"}
          attach="material"
          distort={0.3}
          speed={2}
          roughness={0.4}
          metalness={0.5}
          emissive={isHovered ? "#4299e1" : "#2b6cb0"}
          emissiveIntensity={isHovered ? 0.6 : 0.2}
        />
      </Sphere>
      {/* DNA Strands inside nucleus */}
      {[...Array(8)].map((_, i) => {
        const angle = (i / 8) * Math.PI * 2;
        const x = Math.cos(angle) * 0.6;
        const z = Math.sin(angle) * 0.6;
        return (
          <Line
            key={i}
            points={[
              [x, -0.8, z],
              [x * 0.8, -0.4, z * 0.8],
              [x * 0.6, 0, z * 0.6],
              [x * 0.8, 0.4, z * 0.8],
              [x, 0.8, z],
            ]}
            color={isHovered ? "#60a5fa" : "#3b82f6"}
            lineWidth={isHovered ? 3 : 2}
            transparent
            opacity={0.6}
          />
        );
      })}
    </Float>
  );
};

// RNA Strands Component - Floating RNA strands inside the cell
const RNAStrands = ({ hoveredElement, setHoveredElement, showChanges = false, isHighlighted = false }: Organelle3DProps & { showChanges?: boolean; isHighlighted?: boolean }) => {
  const rnaGroup = useRef<THREE.Group>(null);
  const isHovered = hoveredElement === "rna";

  useFrame(() => {
    if (rnaGroup.current) {
      rnaGroup.current.rotation.y += showChanges ? 0.005 : 0.002;
    }
  });

  // Reduced positions - all inside cell (radius < 2.5, cell membrane at 3)
  const rnaPositions = [
    [1.2, 1.0, 0.3],
    [-1.2, -1.0, -0.3],
    [1.0, -1.2, 0.5],
    [-1.0, 1.2, -0.5],
    [0.3, 1.0, 1.2],
    [-0.3, -1.0, -1.2],
    [1.5, -0.5, 0.8],
    [-1.5, 0.5, -0.8],
  ];

  return (
    <group ref={rnaGroup}>
      {rnaPositions.map((pos, i) => (
        <Float key={i} speed={2 + i * 0.2} rotationIntensity={0.5} floatIntensity={0.5}>
          <group
            position={pos as [number, number, number]}
            onPointerEnter={() => setHoveredElement("rna")}
            onPointerLeave={() => setHoveredElement(null)}
          >
            {/* RNA strand - wavy floating strand */}
            <Line
              points={[
                [0, 0, 0],
                [0.12, 0.2, 0.05],
                [-0.1, 0.4, 0.1],
                [0.1, 0.6, 0.05],
                [-0.12, 0.8, 0.1],
                [0.1, 1.0, 0],
                [-0.08, 1.2, 0.05],
              ]}
              color={(isHovered || showChanges || isHighlighted) ? "#fbbf24" : "#f59e0b"}
              lineWidth={(isHovered || showChanges || isHighlighted) ? 6 : 4}
              transparent
              opacity={(showChanges || isHighlighted) ? 1 : 0.8}
            />
            {/* Additional smaller strand for depth */}
            <Line
              points={[
                [0, 0.1, 0],
                [0.08, 0.3, -0.05],
                [-0.08, 0.5, -0.1],
                [0.08, 0.7, -0.05],
                [-0.08, 0.9, -0.1],
                [0.06, 1.1, 0],
              ]}
              color={(isHovered || showChanges || isHighlighted) ? "#fbbf24" : "#f59e0b"}
              lineWidth={(isHovered || showChanges || isHighlighted) ? 4 : 3}
              transparent
              opacity={(showChanges || isHighlighted) ? 0.8 : 0.6}
            />
            {/* Central node/bead */}
            <Sphere args={[(showChanges || isHighlighted) ? 0.15 : 0.1, 8, 8]} position={[0, 0, 0]}>
              <meshStandardMaterial
                color={(isHovered || showChanges || isHighlighted) ? "#fbbf24" : "#f59e0b"}
                emissive={(isHovered || showChanges || isHighlighted) ? "#fbbf24" : "#f59e0b"}
                emissiveIntensity={(isHovered || showChanges || isHighlighted) ? 1.5 : 0.5}
              />
            </Sphere>
            {/* Additional beads along the strand */}
            <Sphere args={[0.06, 6, 6]} position={[0.1, 0.6, 0]}>
              <meshStandardMaterial
                color={(isHovered || showChanges || isHighlighted) ? "#fbbf24" : "#f59e0b"}
                emissive={(isHovered || showChanges || isHighlighted) ? "#fbbf24" : "#f59e0b"}
                emissiveIntensity={(isHovered || showChanges || isHighlighted) ? 1.2 : 0.4}
              />
            </Sphere>
            <Sphere args={[0.06, 6, 6]} position={[-0.1, 1.0, 0]}>
              <meshStandardMaterial
                color={(isHovered || showChanges || isHighlighted) ? "#fbbf24" : "#f59e0b"}
                emissive={(isHovered || showChanges || isHighlighted) ? "#fbbf24" : "#f59e0b"}
                emissiveIntensity={(isHovered || showChanges || isHighlighted) ? 1.2 : 0.4}
              />
            </Sphere>
          </group>
        </Float>
      ))}
    </group>
  );
};

// Proteins
const Proteins = ({ hoveredElement, setHoveredElement, showChanges = false, isHighlighted = false }: Organelle3DProps & { showChanges?: boolean; isHighlighted?: boolean }) => {
  const proteinGroup = useRef<THREE.Group>(null);
  const isHovered = hoveredElement === "protein";

  useFrame(() => {
    if (proteinGroup.current) {
      proteinGroup.current.rotation.y -= showChanges ? 0.003 : 0.0015;
    }
  });

  // Reduced positions - all inside cell (radius < 2.5, cell membrane at 3)
  const proteinPositions = [
    [1.5, 0.3, 0.6],
    [-1.5, -0.3, -0.6],
    [0.6, 1.5, -0.3],
    [-0.6, -1.5, 0.3],
    [1.2, -1.2, 1.0],
    [-1.2, 1.2, -1.0],
    [0, 1.5, 1.2],
    [0, -1.5, -1.2],
    [1.4, 0.6, -0.6],
    [-1.4, -0.6, 0.6],
  ];

  return (
    <group ref={proteinGroup}>
      {proteinPositions.map((pos, i) => (
        <Float key={i} speed={1.5 + i * 0.15} rotationIntensity={0.3} floatIntensity={0.4}>
          <group
            position={pos as [number, number, number]}
            onPointerEnter={() => setHoveredElement("protein")}
            onPointerLeave={() => setHoveredElement(null)}
          >
            {/* Y-shaped protein */}
            <mesh>
              <sphereGeometry args={[(showChanges || isHighlighted) ? 0.18 : 0.12, 8, 8]} />
              <meshStandardMaterial
                color={(isHovered || showChanges || isHighlighted) ? "#f472b6" : "#ec4899"}
                emissive={(isHovered || showChanges || isHighlighted) ? "#f472b6" : "#ec4899"}
                emissiveIntensity={(isHovered || showChanges || isHighlighted) ? 1.2 : 0.3}
              />
            </mesh>
            <mesh position={[0.08, 0.08, 0]}>
              <sphereGeometry args={[(showChanges || isHighlighted) ? 0.13 : 0.08, 8, 8]} />
              <meshStandardMaterial
                color={(isHovered || showChanges || isHighlighted) ? "#f472b6" : "#ec4899"}
                emissive={(isHovered || showChanges || isHighlighted) ? "#f472b6" : "#ec4899"}
                emissiveIntensity={(isHovered || showChanges || isHighlighted) ? 1.2 : 0.3}
              />
            </mesh>
            <mesh position={[-0.08, 0.08, 0]}>
              <sphereGeometry args={[(showChanges || isHighlighted) ? 0.13 : 0.08, 8, 8]} />
              <meshStandardMaterial
                color={(isHovered || showChanges || isHighlighted) ? "#f472b6" : "#ec4899"}
                emissive={(isHovered || showChanges || isHighlighted) ? "#f472b6" : "#ec4899"}
                emissiveIntensity={(isHovered || showChanges || isHighlighted) ? 1.2 : 0.3}
              />
            </mesh>
          </group>
        </Float>
      ))}
    </group>
  );
};

// Drug Tablet Component
const DrugTablet = ({ isInjecting, injectionProgress, keepVisible = false }: InjectionProps) => {
  const tabletRef = useRef<THREE.Group>(null);
  const [position, setPosition] = useState<[number, number, number]>([6, 4, 4]);
  const [finalPosition, setFinalPosition] = useState<[number, number, number] | null>(null);

  // Initialize final position if keepVisible is true
  useEffect(() => {
    if (keepVisible && !finalPosition) {
      // Position in cytoplasm (not at nucleus center)
      setFinalPosition([1.5, 1.2, 0.8]);
      setPosition([1.5, 1.2, 0.8]);
    }
  }, [keepVisible, finalPosition]);

  useFrame(() => {
    if (isInjecting && injectionProgress !== undefined) {
      const startPos: [number, number, number] = [6, 4, 4]; // Start from outside
      const endPos: [number, number, number] = [1.5, 1.2, 0.8]; // End in cytoplasm (not at nucleus)
      
      const newX = startPos[0] + (endPos[0] - startPos[0]) * injectionProgress;
      const newY = startPos[1] + (endPos[1] - startPos[1]) * injectionProgress;
      const newZ = startPos[2] + (endPos[2] - startPos[2]) * injectionProgress;
      
      setPosition([newX, newY, newZ]);
      
      if (tabletRef.current) {
        tabletRef.current.rotation.y += 0.03;
        tabletRef.current.rotation.x += 0.02;
      }
      
      // Save final position when injection completes
      if (injectionProgress >= 1) {
        setFinalPosition([1.5, 1.2, 0.8]);
      }
    } else if (finalPosition || keepVisible) {
      // Use saved final position or keepVisible position and let Float handle the floating
      const targetPos = finalPosition || [1.5, 1.2, 0.8];
      setPosition(targetPos);
      if (tabletRef.current) {
        tabletRef.current.rotation.y += 0.02;
      }
    }
  });

  // Show if injecting, has final position, or should be kept visible
  if (!isInjecting && !finalPosition && !keepVisible) return null;

  return (
    <Float speed={1.5} rotationIntensity={0.3} floatIntensity={0.4}>
      <group ref={tabletRef} position={position}>
        {/* Tablet pill shape - larger and more visible */}
        <mesh>
          <cylinderGeometry args={[0.25, 0.25, 0.15, 32]} />
          <meshStandardMaterial
            color="#3b82f6"
            emissive="#3b82f6"
            emissiveIntensity={0.8}
            metalness={0.5}
          />
        </mesh>
        {/* Outer glow effect */}
        <mesh>
          <cylinderGeometry args={[0.35, 0.35, 0.18, 32]} />
          <meshBasicMaterial
            color="#3b82f6"
            transparent
            opacity={0.4}
            wireframe
          />
        </mesh>
        {/* Pulsing glow sphere */}
        <Sphere args={[0.4, 16, 16]}>
          <meshBasicMaterial
            color="#3b82f6"
            transparent
            opacity={0.3}
          />
        </Sphere>
      </group>
    </Float>
  );
};

// CRISPR-Cas9 Complex
const CRISPRComplex = ({ isInjecting, injectionProgress, keepVisible = false }: InjectionProps) => {
  const crisprRef = useRef<THREE.Group>(null);
  const [position, setPosition] = useState<[number, number, number]>([-6, 4, 4]);
  const [finalPosition, setFinalPosition] = useState<[number, number, number] | null>(null);

  // Initialize final position if keepVisible is true
  useEffect(() => {
    if (keepVisible && !finalPosition) {
      // Position in cytoplasm (not at nucleus center)
      setFinalPosition([-1.5, 1.0, -0.8]);
      setPosition([-1.5, 1.0, -0.8]);
    }
  }, [keepVisible, finalPosition]);

  useFrame(() => {
    if (isInjecting && injectionProgress !== undefined) {
      // Move from outside to cytoplasm
      const startPos: [number, number, number] = [-6, 4, 4]; // Start from outside
      const endPos: [number, number, number] = [-1.5, 1.0, -0.8]; // End in cytoplasm (not at nucleus)
      
      const newX = startPos[0] + (endPos[0] - startPos[0]) * injectionProgress;
      const newY = startPos[1] + (endPos[1] - startPos[1]) * injectionProgress;
      const newZ = startPos[2] + (endPos[2] - startPos[2]) * injectionProgress;
      
      setPosition([newX, newY, newZ]);
      
      if (crisprRef.current) {
        crisprRef.current.rotation.y += 0.05;
      }
      
      // Save final position when injection completes
      if (injectionProgress >= 1) {
        setFinalPosition([-1.5, 1.0, -0.8]);
      }
    } else if (finalPosition || keepVisible) {
      // Use saved final position or keepVisible position and let Float handle the floating
      const targetPos = finalPosition || [-1.5, 1.0, -0.8];
      setPosition(targetPos);
      if (crisprRef.current) {
        crisprRef.current.rotation.y += 0.03;
      }
    }
  });

  // Show if injecting, has final position, or should be kept visible
  if (!isInjecting && !finalPosition && !keepVisible) return null;

  return (
    <Float speed={1.5} rotationIntensity={0.3} floatIntensity={0.4}>
      <group ref={crisprRef} position={position}>
        {/* Cas9 protein - larger core structure */}
        <mesh>
          <sphereGeometry args={[0.35, 16, 16]} />
          <meshStandardMaterial
            color="#9333ea"
            emissive="#9333ea"
            emissiveIntensity={0.8}
            metalness={0.6}
          />
        </mesh>
        <mesh position={[0.2, 0.15, 0]}>
          <sphereGeometry args={[0.2, 12, 12]} />
          <meshStandardMaterial
            color="#7c3aed"
            emissive="#7c3aed"
            emissiveIntensity={0.6}
          />
        </mesh>
        <mesh position={[-0.2, -0.15, 0]}>
          <sphereGeometry args={[0.2, 12, 12]} />
          <meshStandardMaterial
            color="#7c3aed"
            emissive="#7c3aed"
            emissiveIntensity={0.6}
          />
        </mesh>
        
        {/* sgRNA - guide strand */}
        <Line
          points={[
            [0.25, 0, 0],
            [0.4, 0.15, 0.15],
            [0.5, -0.1, 0.2],
            [0.55, 0.1, 0.25],
            [0.6, -0.15, 0.3],
          ]}
          color="#10b981"
          lineWidth={5}
          transparent
          opacity={0.9}
        />
        
        {/* Outer glow effect */}
        <Sphere args={[0.5, 16, 16]}>
          <meshBasicMaterial
            color="#a855f7"
            transparent
            opacity={0.3}
          />
        </Sphere>
        {/* Wireframe glow */}
        <Sphere args={[0.55, 16, 16]}>
          <meshBasicMaterial
            color="#a855f7"
            transparent
            opacity={0.2}
            wireframe
          />
        </Sphere>
      </group>
    </Float>
  );
};

// Surface Proteins - Membrane-bound proteins
const SurfaceProteins = ({ hoveredElement, setHoveredElement, isHighlighted = false }: Organelle3DProps & { isHighlighted?: boolean }) => {
  const surfaceProteinGroup = useRef<THREE.Group>(null);
  const isHovered = hoveredElement === "protein";

  useFrame(() => {
    if (surfaceProteinGroup.current) {
      surfaceProteinGroup.current.rotation.y += 0.0003;
    }
  });

  // Positions on cell membrane surface (radius ~3)
  // Using spherical coordinates to place them on the surface
  const surfaceProteinPositions = [
    [2.8, 0.5, 0.8],      // Front-right
    [-2.8, -0.5, -0.8],   // Back-left
    [0.5, 2.8, 0.6],      // Top-right
    [-0.5, -2.8, -0.6],   // Bottom-left
    [1.5, 1.5, 2.4],      // Front-top-right
    [-1.5, -1.5, -2.4],   // Back-bottom-left
    [2.2, -1.2, 1.5],     // Front-bottom-right
    [-2.2, 1.2, -1.5],   // Back-top-left
  ];

  return (
    <group ref={surfaceProteinGroup}>
      {surfaceProteinPositions.map((pos, i) => {
        // Normalize to ensure proteins are exactly on the surface
        const [x, y, z] = pos;
        const distance = Math.sqrt(x * x + y * y + z * z);
        const normalizedPos: [number, number, number] = [
          (x / distance) * 3,
          (y / distance) * 3,
          (z / distance) * 3,
        ];

        // Calculate rotation to point protein inward (toward cell center)
        const lookAtCenter = new THREE.Vector3(0, 0, 0);
        const proteinPos = new THREE.Vector3(...normalizedPos);
        const direction = lookAtCenter.clone().sub(proteinPos).normalize();
        const quaternion = new THREE.Quaternion();
        quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), direction);

        return (
          <group key={i} position={normalizedPos} quaternion={quaternion}>
            <group
              onPointerEnter={() => setHoveredElement("protein")}
              onPointerLeave={() => setHoveredElement(null)}
            >
              {/* Membrane protein - transmembrane structure */}
              {/* Extracellular domain (outside) */}
              <mesh position={[0, 0, 0.15]}>
                <sphereGeometry args={[(isHovered || isHighlighted) ? 0.12 : 0.08, 8, 8]} />
                <meshStandardMaterial
                  color={(isHovered || isHighlighted) ? "#f472b6" : "#ec4899"}
                  emissive={(isHovered || isHighlighted) ? "#f472b6" : "#ec4899"}
                  emissiveIntensity={(isHovered || isHighlighted) ? 1.0 : 0.4}
                />
              </mesh>
              {/* Transmembrane domain - cylinder through membrane */}
              <mesh>
                <cylinderGeometry args={[0.04, 0.04, 0.3, 8]} />
                <meshStandardMaterial
                  color={(isHovered || isHighlighted) ? "#f472b6" : "#ec4899"}
                  emissive={(isHovered || isHighlighted) ? "#f472b6" : "#ec4899"}
                  emissiveIntensity={(isHovered || isHighlighted) ? 0.8 : 0.3}
                />
              </mesh>
              {/* Intracellular domain (inside) */}
              <mesh position={[0, 0, -0.15]}>
                <sphereGeometry args={[(isHovered || isHighlighted) ? 0.1 : 0.07, 8, 8]} />
                <meshStandardMaterial
                  color={(isHovered || isHighlighted) ? "#f472b6" : "#ec4899"}
                  emissive={(isHovered || isHighlighted) ? "#f472b6" : "#ec4899"}
                  emissiveIntensity={(isHovered || isHighlighted) ? 1.0 : 0.4}
                />
              </mesh>
            </group>
          </group>
        );
      })}
    </group>
  );
};

// Cell Membrane
const CellMembrane = () => {
  const membraneRef = useRef<THREE.Mesh>(null);

  useFrame(() => {
    if (membraneRef.current) {
      membraneRef.current.rotation.y += 0.0005;
    }
  });

  return (
    <Sphere ref={membraneRef} args={[3, 64, 64]}>
      <meshPhysicalMaterial
        color="#e5e7eb"
        transparent
        opacity={0.15}
        roughness={0.2}
        metalness={0.1}
        transmission={0.9}
        thickness={0.5}
        wireframe={false}
      />
    </Sphere>
  );
};

// Main Cell Component
export const Cell3D = ({ 
  hoveredElement, 
  setHoveredElement, 
  isInjecting, 
  onInjectionComplete,
  showRNAChanges = false,
  showProteinChanges = false,
  perturbationType = "gene",
  selectedNode = null,
  hasBothPerturbations = false,
  hasGeneInjected = false,
  hasDrugInjected = false
}: Cell3DProps) => {
  const [injectionProgress, setInjectionProgress] = useState(0);
  const [hasInjected, setHasInjected] = useState(false);

  useEffect(() => {
    if (isInjecting) {
      setInjectionProgress(0);
      const duration = 3000; // 3 seconds
      const startTime = Date.now();
      
      const animate = () => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        setInjectionProgress(progress);
        
        if (progress < 1) {
          requestAnimationFrame(animate);
        } else {
          setHasInjected(true);
          setTimeout(() => {
            onInjectionComplete();
          }, 500);
        }
      };
      
      requestAnimationFrame(animate);
    } else {
      setInjectionProgress(0);
    }
  }, [isInjecting, onInjectionComplete]);

  return (
    <div className="w-full h-[600px] rounded-xl overflow-hidden bg-gradient-to-br from-background to-secondary/30">
      <Canvas camera={{ position: [0, 0, 8], fov: 50 }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        <pointLight position={[-10, -10, -10]} intensity={0.5} color="#4299e1" />
        <spotLight position={[0, 10, 0]} angle={0.3} penumbra={1} intensity={0.5} />
        <pointLight position={[-5, 3, 3]} intensity={0.8} color="#a855f7" />

        <CellMembrane />
        <Nucleus hoveredElement={hoveredElement} setHoveredElement={setHoveredElement} />
        
        {/* RNA Strands - Floating strands inside cell */}
        <RNAStrands 
          hoveredElement={hoveredElement} 
          setHoveredElement={setHoveredElement} 
          showChanges={showRNAChanges}
          isHighlighted={selectedNode === "rna"}
        />
        <Proteins 
          hoveredElement={hoveredElement} 
          setHoveredElement={setHoveredElement} 
          showChanges={showProteinChanges}
          isHighlighted={selectedNode === "protein"}
        />
        
        {/* Surface Proteins - Membrane-bound proteins */}
        <SurfaceProteins 
          hoveredElement={hoveredElement} 
          setHoveredElement={setHoveredElement} 
          isHighlighted={selectedNode === "protein"}
        />
        
        {/* Perturbation System */}
        {/* Show CRISPR if injecting gene, has both, or gene was previously injected */}
        {(hasBothPerturbations || perturbationType === "gene" || hasGeneInjected) && (
          <CRISPRComplex 
            isInjecting={isInjecting && (hasBothPerturbations || perturbationType === "gene")} 
            injectionProgress={injectionProgress} 
            keepVisible={hasGeneInjected}
          />
        )}
        {/* Show Drug if injecting drug, has both, or drug was previously injected */}
        {(hasBothPerturbations || perturbationType === "drug" || hasDrugInjected) && (
          <DrugTablet 
            isInjecting={isInjecting && (hasBothPerturbations || perturbationType === "drug")} 
            injectionProgress={injectionProgress} 
            keepVisible={hasDrugInjected}
          />
        )}

        <OrbitControls
          enablePan={false}
          enableZoom={true}
          minDistance={5}
          maxDistance={12}
          autoRotate={!isInjecting}
          autoRotateSpeed={0.5}
        />
      </Canvas>
    </div>
  );
};
