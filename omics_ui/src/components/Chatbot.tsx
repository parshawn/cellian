import { useState, useRef, useEffect } from "react";
import { Send, MessageSquare, X, Bot, AlertCircle, CheckCircle2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { extractPerturbationInfo, PerturbationInfo } from "@/utils/perturbationExtractor";
import { processQuery } from "@/utils/api";

interface Message {
  id: string;
  text: string;
  sender: "user" | "bot";
  timestamp: Date;
  perturbationInfo?: PerturbationInfo;
}

interface ChatbotProps {
  isActive: boolean;
  onQuerySubmit: (query: string, perturbationInfo?: PerturbationInfo) => void;
  onInjectionTrigger?: (type: "gene" | "drug" | "both", perturbationInfo?: PerturbationInfo) => void;
}

export const Chatbot = ({ isActive, onQuerySubmit, onInjectionTrigger }: ChatbotProps) => {
  const [isOpen, setIsOpen] = useState(true);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      text: "Hello! I'm your AI Hypothesis Assistant. Ask me about CRISPR perturbations, drug effects, or generate hypotheses about cellular processes.",
      sender: "bot",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const shouldScrollRef = useRef(true);

  useEffect(() => {
    // Only auto-scroll if we should (not when user is reading old messages)
    if (shouldScrollRef.current && messagesEndRef.current) {
      // Use setTimeout to ensure DOM is updated
      setTimeout(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }, 100);
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userInput = input;
    const userMessage: Message = {
      id: Date.now().toString(),
      text: userInput,
      sender: "user",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    // Reset scroll behavior when sending
    shouldScrollRef.current = true;

    // Show loading message
    const loadingMessage: Message = {
      id: (Date.now() + 0.1).toString(),
      text: "Processing your query with AI...",
      sender: "bot",
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, loadingMessage]);

    try {
      // Call backend API to process query using Gemini
      const perturbationInfo = await processQuery(userInput);
      
      // Update user message with perturbation info
      setMessages((prev) => 
        prev.map(msg => 
          msg.id === userMessage.id 
            ? { ...msg, perturbationInfo: perturbationInfo as any }
            : msg
        )
      );

      // Remove loading message and add response
      setMessages((prev) => prev.filter(msg => msg.id !== loadingMessage.id));

      // Validation feedback
      if (perturbationInfo.target) {
        const validationMsg = perturbationInfo.confidence >= 0.8 
          ? `✅ Validated: ${perturbationInfo.target} (${perturbationInfo.type})`
          : `⚠️ Low confidence: ${perturbationInfo.target} (${perturbationInfo.type})`;
        
        const validationMessage: Message = {
          id: (Date.now() + 0.5).toString(),
          text: validationMsg,
          sender: "bot",
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, validationMessage]);
      }

      // Determine perturbation type from extracted info
      const isGene = perturbationInfo.type === "KO" || perturbationInfo.type === "KD" || perturbationInfo.type === "OE" || 
                     (perturbationInfo.target && /^[A-Z]/.test(perturbationInfo.target));
      const isDrug = perturbationInfo.type === "drug" || 
                     (perturbationInfo.target && /^[a-z]/.test(perturbationInfo.target));
      
      if (isGene && isDrug) {
        // Both gene and drug
        onInjectionTrigger?.("both", perturbationInfo as any);
        const botMessage: Message = {
          id: (Date.now() + 1).toString(),
          text: `Injecting both CRISPR-Cas9 gene perturbation (${perturbationInfo.target}) and drug compound into the cell...`,
          sender: "bot",
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, botMessage]);
      } else if (isGene && perturbationInfo.target) {
        onInjectionTrigger?.("gene", perturbationInfo as any);
        const botMessage: Message = {
          id: (Date.now() + 1).toString(),
          text: `Injecting CRISPR-Cas9 gene perturbation (${perturbationInfo.target} ${perturbationInfo.type}) into the cell...`,
          sender: "bot",
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, botMessage]);
      } else if (isDrug && perturbationInfo.target) {
        onInjectionTrigger?.("drug", perturbationInfo as any);
        const botMessage: Message = {
          id: (Date.now() + 1).toString(),
          text: `Injecting drug compound (${perturbationInfo.target}) into the cell...`,
          sender: "bot",
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, botMessage]);
      } else {
        // Regular query - ask for clarification
        onQuerySubmit(userInput, perturbationInfo as any);
        const botMessage: Message = {
          id: (Date.now() + 1).toString(),
          text: perturbationInfo.target 
            ? `I detected a perturbation: ${perturbationInfo.target}. Could you clarify what specific question you'd like me to answer about this perturbation?`
            : "Thank you for your query! Could you provide more details about what you'd like to analyze?",
          sender: "bot",
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, botMessage]);
      }
    } catch (error) {
      // Remove loading message
      setMessages((prev) => prev.filter(msg => msg.id !== loadingMessage.id));
      
      // Fallback to local extraction if API fails
      const perturbationInfo = extractPerturbationInfo(userInput);
      setMessages((prev) => 
        prev.map(msg => 
          msg.id === userMessage.id 
            ? { ...msg, perturbationInfo }
            : msg
        )
      );

      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: "⚠️ Backend API unavailable. Using local extraction. Please ensure the backend server is running.",
        sender: "bot",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);

      // Continue with local extraction logic
      const isGene = perturbationInfo.type === "KO" || perturbationInfo.type === "KD" || perturbationInfo.type === "OE";
      const isDrug = perturbationInfo.type === "drug";
      
      if (isGene && perturbationInfo.target) {
        onInjectionTrigger?.("gene", perturbationInfo);
      } else if (isDrug && perturbationInfo.target) {
        onInjectionTrigger?.("drug", perturbationInfo);
      } else {
        onQuerySubmit(userInput, perturbationInfo);
      }
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <Card className="bg-card/80 backdrop-blur rounded-xl border-2 border-border shadow-lg overflow-hidden flex flex-col h-full">
      <div className="flex items-center justify-between p-4 border-b border-border bg-gradient-to-r from-dna/10 to-protein/10">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-full bg-gradient-to-r from-dna to-protein flex items-center justify-center">
            <Bot className="w-4 h-4 text-white" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-foreground">AI Hypothesis Assistant</h3>
            <p className="text-xs text-muted-foreground">Ask questions about perturbations</p>
          </div>
        </div>
        <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setIsOpen(!isOpen)}>
          <X className="w-4 h-4" />
        </Button>
      </div>

      {isOpen && (
        <>
          <ScrollArea 
            className="flex-1 p-4 min-h-[300px]"
            onScrollCapture={() => {
              // User is scrolling manually, don't auto-scroll
              const scrollContainer = document.querySelector('[data-radix-scroll-area-viewport]');
              if (scrollContainer) {
                const { scrollTop, scrollHeight, clientHeight } = scrollContainer;
                const isAtBottom = scrollHeight - scrollTop - clientHeight < 50;
                shouldScrollRef.current = isAtBottom;
              }
            }}
          >
            <div className="space-y-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[85%] rounded-lg px-3 py-2 ${
                      message.sender === "user"
                        ? "bg-gradient-to-r from-dna to-protein text-white"
                        : "bg-secondary text-foreground border border-border"
                    }`}
                  >
                    <p className="text-sm leading-relaxed">{message.text}</p>
                    <p className="text-xs opacity-70 mt-1">
                      {message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                    </p>
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          </ScrollArea>

          <div className="p-4 border-t border-border bg-secondary/30">
            <div className="flex gap-2">
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type 'gene' or 'drug' to inject, or ask questions..."
                className="flex-1 bg-background"
              />
              <Button onClick={handleSend} disabled={!input.trim()} size="icon" className="bg-gradient-to-r from-dna to-protein hover:opacity-90">
                <Send className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </>
      )}
    </Card>
  );
};
