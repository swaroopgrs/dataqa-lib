"""Event bus implementation for inter-agent communication and coordination."""

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import UUID

from .message_protocols import Message, MessageType, MessagePriority
from .persistence import EventStore
from .routing import MessageRouter
from .security import MessageSecurity

logger = logging.getLogger(__name__)


class EventHandler:
    """Handler for processing events from the event bus."""
    
    def __init__(
        self,
        handler_id: str,
        handler_func: Callable[[Message], Any],
        message_types: Optional[Set[MessageType]] = None,
        priority_filter: Optional[MessagePriority] = None
    ):
        self.handler_id = handler_id
        self.handler_func = handler_func
        self.message_types = message_types or set()
        self.priority_filter = priority_filter
        self.is_active = True
    
    async def handle(self, message: Message) -> Any:
        """Handle a message if it matches the handler criteria."""
        if not self.is_active:
            return None
        
        # Check message type filter
        if self.message_types and message.message_type not in self.message_types:
            return None
        
        # Check priority filter
        if self.priority_filter and message.priority != self.priority_filter:
            return None
        
        try:
            if asyncio.iscoroutinefunction(self.handler_func):
                return await self.handler_func(message)
            else:
                return self.handler_func(message)
        except Exception as e:
            logger.error(f"Error in handler {self.handler_id}: {e}")
            raise
    
    def deactivate(self):
        """Deactivate this handler."""
        self.is_active = False


class EventBus:
    """Central event bus for inter-agent communication and coordination."""
    
    def __init__(
        self,
        event_store: Optional[EventStore] = None,
        message_router: Optional[MessageRouter] = None,
        security: Optional[MessageSecurity] = None,
        max_queue_size: int = 10000
    ):
        self.event_store = event_store
        self.message_router = message_router
        self.security = security
        self.max_queue_size = max_queue_size
        
        # Handler management
        self.handlers: Dict[str, EventHandler] = {}
        self.topic_handlers: Dict[MessageType, List[EventHandler]] = defaultdict(list)
        
        # Message queues and processing
        self.message_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.priority_queues: Dict[MessagePriority, asyncio.Queue] = {
            priority: asyncio.Queue(maxsize=max_queue_size)
            for priority in MessagePriority
        }
        
        # Processing control
        self.is_running = False
        self.processing_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_processed": 0,
            "messages_failed": 0,
            "handlers_registered": 0,
        }
    
    async def start(self):
        """Start the event bus processing."""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_messages())
        logger.info("Event bus started")
    
    async def stop(self):
        """Stop the event bus processing."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Event bus stopped")
    
    def register_handler(
        self,
        handler_id: str,
        handler_func: Callable[[Message], Any],
        message_types: Optional[Set[MessageType]] = None,
        priority_filter: Optional[MessagePriority] = None
    ) -> EventHandler:
        """Register a message handler."""
        handler = EventHandler(
            handler_id=handler_id,
            handler_func=handler_func,
            message_types=message_types,
            priority_filter=priority_filter
        )
        
        self.handlers[handler_id] = handler
        
        # Add to topic-specific handlers
        if message_types:
            for msg_type in message_types:
                self.topic_handlers[msg_type].append(handler)
        else:
            # Handler for all message types
            for msg_type in MessageType:
                self.topic_handlers[msg_type].append(handler)
        
        self.stats["handlers_registered"] += 1
        logger.info(f"Registered handler: {handler_id}")
        return handler
    
    def unregister_handler(self, handler_id: str):
        """Unregister a message handler."""
        if handler_id not in self.handlers:
            return
        
        handler = self.handlers[handler_id]
        handler.deactivate()
        
        # Remove from topic handlers
        for handlers_list in self.topic_handlers.values():
            if handler in handlers_list:
                handlers_list.remove(handler)
        
        del self.handlers[handler_id]
        logger.info(f"Unregistered handler: {handler_id}")
    
    async def publish(self, message: Message) -> bool:
        """Publish a message to the event bus."""
        try:
            # Security check
            if self.security and not await self.security.authorize_message(message):
                logger.warning(f"Message {message.message_id} failed security check")
                return False
            
            # Store event if persistence is enabled
            if self.event_store:
                await self.event_store.store_event(message)
            
            # Route message if router is configured
            if self.message_router:
                routed_messages = await self.message_router.route_message(message)
                for routed_msg in routed_messages:
                    await self._enqueue_message(routed_msg)
            else:
                await self._enqueue_message(message)
            
            self.stats["messages_sent"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish message {message.message_id}: {e}")
            self.stats["messages_failed"] += 1
            return False
    
    async def _enqueue_message(self, message: Message):
        """Enqueue a message for processing."""
        # Use priority queue if available
        if message.priority in self.priority_queues:
            try:
                self.priority_queues[message.priority].put_nowait(message)
            except asyncio.QueueFull:
                logger.warning(f"Priority queue {message.priority} is full, using main queue")
                await self.message_queue.put(message)
        else:
            await self.message_queue.put(message)
    
    async def _process_messages(self):
        """Main message processing loop."""
        while self.is_running:
            try:
                # Process priority messages first
                message = await self._get_next_message()
                if message:
                    await self._handle_message(message)
                    self.stats["messages_processed"] += 1
                else:
                    # No messages available, short sleep
                    await asyncio.sleep(0.01)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                self.stats["messages_failed"] += 1
    
    async def _get_next_message(self) -> Optional[Message]:
        """Get the next message to process, prioritizing by priority level."""
        # Check priority queues in order
        for priority in [MessagePriority.CRITICAL, MessagePriority.HIGH, 
                        MessagePriority.NORMAL, MessagePriority.LOW]:
            queue = self.priority_queues[priority]
            if not queue.empty():
                try:
                    return queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue
        
        # Check main queue
        try:
            return await asyncio.wait_for(self.message_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None
    
    async def _handle_message(self, message: Message):
        """Handle a message by dispatching to registered handlers."""
        handlers = self.topic_handlers.get(message.message_type, [])
        
        if not handlers:
            logger.debug(f"No handlers for message type: {message.message_type}")
            return
        
        # Execute handlers concurrently
        tasks = []
        for handler in handlers:
            if handler.is_active:
                task = asyncio.create_task(handler.handle(message))
                tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log any handler exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Handler {handlers[i].handler_id} failed: {result}")
    
    async def request_response(
        self,
        request_message: Message,
        timeout_seconds: float = 30.0
    ) -> Optional[Message]:
        """Send a request message and wait for a response."""
        response_future = asyncio.Future()
        correlation_id = request_message.message_id
        
        # Register temporary handler for response
        def response_handler(message: Message) -> None:
            if (message.correlation_id == correlation_id and 
                not response_future.done()):
                response_future.set_result(message)
        
        handler_id = f"temp_response_{correlation_id}"
        self.register_handler(
            handler_id=handler_id,
            handler_func=response_handler,
            message_types={MessageType.STATUS_UPDATE}
        )
        
        try:
            # Send request
            await self.publish(request_message)
            
            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=timeout_seconds)
            return response
            
        except asyncio.TimeoutError:
            logger.warning(f"Request {correlation_id} timed out")
            return None
        finally:
            # Clean up temporary handler
            self.unregister_handler(handler_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            **self.stats,
            "active_handlers": len([h for h in self.handlers.values() if h.is_active]),
            "queue_size": self.message_queue.qsize(),
            "priority_queue_sizes": {
                priority.value: queue.qsize() 
                for priority, queue in self.priority_queues.items()
            },
            "is_running": self.is_running,
        }