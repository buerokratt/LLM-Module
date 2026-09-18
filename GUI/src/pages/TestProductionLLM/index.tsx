import { FC, useState, useRef, useEffect, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { Button, FormTextarea, FormSelect } from 'components';
import { useToast } from 'hooks/useToast';
import { useStreamingResponse } from 'hooks/useStreamingResponse';
import { ChoiceButton } from 'services/inference';
import './TestProductionLLM.scss';
import MessageContent from 'components/MessageContent';
import { llmConnectionsQueryKeys } from 'utils/queryKeys';
import { fetchAllLLMConnectionsPaginated } from 'services/llmConnections';


interface Message {
  id: string;
  content: string;
  isUser: boolean;
  timestamp: string;
  hasError?: boolean;
  errorMessage?: string;
  buttons?: ChoiceButton[];
}

const TestProductionLLM: FC = () => {
  const { t } = useTranslation();
  const toast = useToast();
  const [inputMessage, setInputMessage] = useState<string>('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [testLLM, setTestLLM] = useState({
    connectionId: null,
    text: '',
  });

  // Generate a unique channel ID for this session
  const channelId = useMemo(() => `channel-${Math.random().toString(36).substring(2, 15)}`, []);
  const { startStreaming, stopStreaming, isStreaming } = useStreamingResponse(channelId);
  const [selectedConnectionId, setSelectedConnectionId] = useState<string | null>(null);

   // Fetch LLM connections for dropdown - using the working legacy endpoint for now
    const { data: connections, isLoading: isLoadingConnections } = useQuery({
      queryKey: llmConnectionsQueryKeys.list({
        page: 1,
        pageSize: 100, // Get all connections for dropdown
        sorting: 'created_at desc',
      }),
      queryFn: () => fetchAllLLMConnectionsPaginated({
        pageNumber: 1,
        pageSize: 100,
        sortBy: 'created_at desc',
      }),
    });
    // Transform connections data for dropdown
  const connectionOptions = useMemo(
    () =>
      connections?.map((connection: any) => ({
        label: `${connection.llmPlatform} - ${connection.llmModel} (${connection.environment})`,
        value: String(connection.id),
      })) || [],
    [connections]
  );

    const selectedConnection = useMemo(() => {
    return connections?.find((conn: any) => String(conn.id) === selectedConnectionId) || null;
  }, [connections, selectedConnectionId]);

  const handleConnectionChange = (value: string | number) => {
    console.log('Selected connection ID:', value);
    if (isLoading || isStreaming) return;
    setSelectedConnectionId(value ? String(value) : null);
  };
  
  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Cleanup incomplete messages on unmount if streaming is active
  useEffect(() => {
    return () => {
      if (isStreaming) {
        stopStreaming();
        // Remove incomplete bot messages on unmount
        setMessages(prev => prev.filter(msg => msg.isUser || !msg.content.trim() === false));
      }
    };
  }, [isStreaming, stopStreaming]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) {
      toast.open({
        type: 'warning',
        title: t('testProductionLLM.warningTitle'),
        message: t('testProductionLLM.emptyMessageWarning'),
      });
      return;
    }

    const userMessageText = inputMessage.trim();
    
    // Add user message
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      content: userMessageText,
      isUser: true,
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    // Create bot message ID
    const botMessageId = `bot-${Date.now()}`;

    // Prepare conversation history (exclude the current user message)
    const conversationHistory = messages.map(msg => ({
      authorRole: msg.isUser ? 'user' : 'bot',
      message: msg.content,
      timestamp: msg.timestamp,
    }));

    const streamingOptions = {
      authorId: 'test-user-456',
      conversationHistory,
      url: 'opensearch-dashboard-test',
      environment: selectedConnection?.environment || 'production',
      connection_id: selectedConnection?.vaultUuid || undefined,
    };

    // Callbacks for streaming
    const onToken = (token: string) => {
      console.log('[Component] Received token:', token);
      
      setMessages(prev => {
        // Find the bot message
        const botMsgIndex = prev.findIndex(msg => msg.id === botMessageId);
        
        if (botMsgIndex === -1) {
          // First token - add the bot message
          console.log('[Component] Adding bot message with first token');
          return [
            ...prev,
            {
              id: botMessageId,
              content: token,
              isUser: false,
              timestamp: new Date().toISOString(),
            }
          ];
        } else {
          // Append token to existing message
          console.log('[Component] Appending token to existing message');
          const updated = [...prev];
          updated[botMsgIndex] = {
            ...updated[botMsgIndex],
            content: updated[botMsgIndex].content + token,
          };
          return updated;
        }
      });
    };

    const onButtons = (buttons: ChoiceButton[]) => {
      setMessages(prev => {
        const botMsgIndex = prev.findIndex(msg => msg.id === botMessageId);
        if (botMsgIndex === -1) return prev;
        const updated = [...prev];
        updated[botMsgIndex] = { ...updated[botMsgIndex], buttons };
        return updated;
      });
    };

    const onComplete = () => {
      console.log('[Component] Stream completed');
      // Always reset loading state on completion
      setIsLoading(false);
    };

    const onError = (error: string) => {
      console.error('[Component] Stream error:', error);
      // Always reset loading state on error
      setIsLoading(false);
      
      // Handle incomplete bot message
      setMessages(prev => {
        const botMsgIndex = prev.findIndex(msg => msg.id === botMessageId);
        
        if (botMsgIndex !== -1) {
          const botMessage = prev[botMsgIndex];
          
          // If the bot message has content, mark it as errored
          if (botMessage.content.trim()) {
            const updated = [...prev];
            updated[botMsgIndex] = {
              ...botMessage,
              hasError: true,
              errorMessage: error,
            };
            return updated;
          } else {
            // If no content, remove the empty bot message
            return prev.filter(msg => msg.id !== botMessageId);
          }
        }
        
        return prev;
      });
      
      toast.open({
        type: 'error',
        title: t('testProductionLLM.streamingErrorTitle'),
        message: error,
      });
    };

    // Start streaming
    try {
      await startStreaming(userMessageText, streamingOptions, onToken, onComplete, onError, onButtons);
    } catch (error) {
      console.error('[Component] Failed to start streaming:', error);
      // Reset loading state if streaming fails to start
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleButtonClick = async (title: string, payload: string) => {
    if (isLoading || isStreaming) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      content: title,
      isUser: true,
      timestamp: new Date().toISOString(),
    };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    const botMessageId = `bot-${Date.now()}`;
    const conversationHistory = messages.map(msg => ({
      authorRole: msg.isUser ? 'user' : 'bot',
      message: msg.content,
      timestamp: msg.timestamp,
    }));
    const streamingOptions = {
      authorId: 'test-user-456',
      conversationHistory,
      url: 'opensearch-dashboard-test',
    };

    const onToken = (token: string) => {
      setMessages(prev => {
        const idx = prev.findIndex(m => m.id === botMessageId);
        if (idx === -1) return [...prev, { id: botMessageId, content: token, isUser: false, timestamp: new Date().toISOString() }];
        const updated = [...prev];
        updated[idx] = { ...updated[idx], content: updated[idx].content + token };
        return updated;
      });
    };
    const onButtons = (buttons: ChoiceButton[]) => {
      setMessages(prev => {
        const idx = prev.findIndex(m => m.id === botMessageId);
        if (idx === -1) return prev;
        const updated = [...prev];
        updated[idx] = { ...updated[idx], buttons };
        return updated;
      });
    };
    const onComplete = () => setIsLoading(false);
    const onError = (error: string) => {
      setIsLoading(false);
      toast.open({ type: 'error', title: t('testProductionLLM.streamingErrorTitle'), message: error });
    };

    try {
      await startStreaming(payload, streamingOptions, onToken, onComplete, onError, onButtons);
    } catch (error) {
      console.error('[Component] Failed to start streaming for button click:', error);
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    stopStreaming();
    toast.open({
      type: 'info',
      title: t('testProductionLLM.chatClearedTitle'),
      message: t('testProductionLLM.chatClearedMessage'),
    });
  };

  return (
    <div>
      <div className="test-production-llm">
        <div className="test-production-llm__header">
          <h1>{t('testModels.title')}</h1>
          <Button onClick={clearChat} appearance="secondary">
            {t('testProductionLLM.clearChat')}
          </Button>
        </div>
           <div className="llm-connection-section">
            <p>{t('testModels.llmConnectionLabel') || 'LLM Connection'}</p>
            <div className="llm-connection-controls">
              <FormSelect
                label=""
                name="connectionId"
                options={connectionOptions}
                placeholder={t('testModels.selectConnectionPlaceholder') || 'Select LLM Connection'}
                onSelectionChange={(selection) => {
                  handleConnectionChange(selection?.value as string);
                }}
                defaultValue={selectedConnectionId ?? undefined}
                disabled={isLoading || isStreaming}
              />
            </div>
          </div>

        <div className="test-production-llm__chat-container">
          <div className="test-production-llm__messages">
            {messages.length === 0 && (
              <div className="test-production-llm__welcome">
                <p>{t('testProductionLLM.welcomeTitle')}</p>
                <p>{t('testProductionLLM.welcomeSubtitle')}</p>
              </div>
            )}
            
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`test-production-llm__message ${
                  msg.isUser ? 'test-production-llm__message--user' : 'test-production-llm__message--bot'
                } ${
                  msg.hasError ? 'test-production-llm__message--error' : ''
                }`}
              >
                <div className="test-production-llm__message-content">
                  <MessageContent content={msg.content} />
                  {!msg.isUser && msg.buttons && msg.buttons.length > 0 && (
                    <div className="mcq-buttons">
                      {msg.buttons.map((btn) => (
                        <Button
                          key={btn.payload}
                          onClick={() => handleButtonClick(btn.title, btn.payload)}
                          disabled={isLoading || isStreaming}
                          appearance="secondary"
                        >
                          {btn.title}
                        </Button>
                      ))}
                    </div>
                  )}
                  {msg.hasError && (
                    <div className="test-production-llm__message-error">
                      <span className="test-production-llm__message-error-icon">⚠️</span>
                      <span className="test-production-llm__message-error-text">
                        {t('testProductionLLM.incompleteMessageError', { defaultValue: 'This message is incomplete due to an error' })}
                        {msg.errorMessage && `: ${msg.errorMessage}`}
                      </span>
                    </div>
                  )}
                </div>
                <div className="test-production-llm__message-timestamp">
                  {new Date(msg.timestamp).toLocaleTimeString()}
                </div>
              </div>
            ))}
            
            {isLoading && (messages.length === 0 || messages[messages.length - 1].isUser) && (
              <div className="test-production-llm__message test-production-llm__message--bot">
                <div className="test-production-llm__message-content">
                  <div className="test-production-llm__typing">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          <div className="test-production-llm__input-area">
            <FormTextarea
              label={t('testProductionLLM.messageLabel')}
              name="message"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder={t('testProductionLLM.messagePlaceholder')??""}
              hideLabel
              maxRows={4}
              disabled={isLoading || isStreaming}
            />
            <Button
              onClick={handleSendMessage}
              disabled={isLoading || isStreaming || !inputMessage.trim()}
              className="test-production-llm__send-button"
            >
              {isLoading || isStreaming ? t('testProductionLLM.sendingButton') : t('testProductionLLM.sendButton')}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TestProductionLLM;