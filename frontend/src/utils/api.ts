/**
 * API utility functions for communicating with the backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export interface PerturbationInfo {
  target: string | null;
  type: string;
  confidence: number;
  original_query?: string;
  originalQuery?: string;
  target2?: string | null;
  type2?: string;
  is_comparison?: boolean;
  isComparison?: boolean;
  [key: string]: any;
}

export interface LogEntry {
  type: string;
  message: string;
  timestamp?: number;
}

export interface WorkflowStatus {
  status: "pending" | "running" | "completed" | "error";
  current_step: string;
  progress: number;
  results?: any;
  error?: string;
  logs?: LogEntry[];
  pipeline_stage?: "perturbation" | "rna" | "protein" | "analysis" | "completed" | "error";
}

/**
 * Process user query using LLM API
 */
export async function processQuery(query: string): Promise<PerturbationInfo> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/query/process`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
    });

    if (!response.ok) {
      throw new Error(`Failed to process query: ${response.statusText}`);
    }

    const data = await response.json();
    return data.perturbation_info;
  } catch (error) {
    console.error("API error:", error);
    throw error;
  }
}

/**
 * Start perturbation workflow
 */
export async function startWorkflow(
  perturbationInfo: PerturbationInfo,
  condition: string,
  perturbationType: "gene" | "drug"
): Promise<{ workflow_id: string; status: string }> {
  const response = await fetch(`${API_BASE_URL}/api/workflow/start`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      perturbation_info: perturbationInfo,
      condition,
      perturbation_type: perturbationType,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to start workflow: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get workflow status
 */
export async function getWorkflowStatus(workflowId: string): Promise<WorkflowStatus> {
  const response = await fetch(`${API_BASE_URL}/api/workflow/${workflowId}/status`);

  if (!response.ok) {
    throw new Error(`Failed to get workflow status: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Stream workflow results using Server-Sent Events
 */
export function streamWorkflowResults(
  workflowId: string,
  onUpdate: (status: WorkflowStatus) => void,
  onError?: (error: Error) => void
): () => void {
  try {
    const eventSource = new EventSource(`${API_BASE_URL}/api/workflow/${workflowId}/stream`);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onUpdate(data);
        
        // Close if completed or error
        if (data.status === "completed" || data.status === "error") {
          eventSource.close();
        }
      } catch (error) {
        console.warn("Error parsing SSE message:", error);
        if (onError) {
          onError(error as Error);
        }
      }
    };

    eventSource.onerror = (error) => {
      console.warn("EventSource error:", error);
      if (onError) {
        onError(new Error("EventSource connection failed"));
      }
      eventSource.close();
    };

    // Return cleanup function
    return () => {
      try {
        eventSource.close();
      } catch (e) {
        // Ignore cleanup errors
      }
    };
  } catch (error) {
    console.error("Failed to create EventSource:", error);
    if (onError) {
      onError(error as Error);
    }
    return () => {};
  }
}
