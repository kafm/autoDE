from typing import Callable, Dict, List, Any
from functools import partial
from dataclasses import dataclass

@dataclass
class Subscription:
    callback: Callable[[Any], Any]
    topic: str
    ref: int

class PubSub:
    def __init__(self):
        self._subscribers:Dict[str, List[Subscription]] = {}
        self._ref = 0

    def clean_up(self):
        self._subscribers = {}
        self._ref = 0

    def publish(self, topic: str, payload: Any):
        subscribers = self._subscribers.get(topic, [])
        for subscription in subscribers:
            subscription.callback(payload)

    def subscribe(self, topic: str, callback: Callable[[Any], Any])->Callable[[], Any]:
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        subscription = Subscription(
                callback=callback,
                topic=topic,
                ref=self._generate_ref()
        )
        self._subscribers[topic].append(subscription)
        return partial(self._unsubscribe, subscription=subscription)

    def _unsubscribe(self, subscription: Subscription):
        subscribers = self._subscribers.get(subscription.topic, [])
        subscription in subscribers and subscribers.remove(subscription)
      
    def _generate_ref(self)->int:
        self._ref += 1
        return self._ref
    
GENERATION_START_EVT = "g0"
GENERATION_ENDED_EVT = "g1"
UPDATED_START_EVT = "u0"
UPDATED_END_EVT = "u1"

OPTIMIZATION_END_EVT = "o1"