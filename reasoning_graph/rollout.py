from __future__ import annotations

import asyncio
import aiohttp
import random
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator, Any
from urllib.parse import urlparse, urlunparse


class SimpleLogger:
    def debug(self, msg): pass
    def info(self, msg): print(f"[INFO] {msg}")
    def error(self, msg): print(f"[ERROR] {msg}")
    def trace(self, msg): pass


logger = SimpleLogger()


class AsyncConsulLoadBalanceProxy:
    @dataclass
    class ServiceInfo:
        address: str
        lucky_number: float
        num_connections: int = 0
        recent_failed_count: int = 0
        recent_failed_time: float = 0
        is_available: bool = True

    def __init__(
        self,
        service_name: str,
        max_connections_per_service: int = 256,
    ) -> None:
        self.service_name = service_name
        self.max_connections_per_service = max_connections_per_service
        self.services = []
        self.next_update_time = 0
        self.service_update_lock = asyncio.Lock()

    async def _update_services(self) -> None:
        if self.service_update_lock.locked():
            return

        async with self.service_update_lock:
            addresses = [f"{self.service_name}.app.msh.team:443"]
            addresses_unordered = set(addresses)
            new_services = []
            for service in self.services:
                if service.address in addresses_unordered:
                    service.is_available = True
                    new_services.append(service)
                    addresses_unordered.remove(service.address)
                elif service.num_connections > 0:
                    service.is_available = False
                    new_services.append(service)
            for address in addresses_unordered:
                new_services.append(
                    self.ServiceInfo(
                        address=address,
                        lucky_number=random.random(),
                        is_available=True,
                    ),
                )
            self.services = new_services

            total_connections = sum(service.num_connections for service in self.services)
            logger.debug(f'Found {len(self.services)} available {self.service_name} services. Total connections: {total_connections}')

    def _calculate_service_sadness(
        self, current_time: float, service, address_hint=None
    ) -> float:
        sadness = 0
        if address_hint is not None and service.address in address_hint:
            sadness -= 10000000000
        if not service.is_available:
            sadness += 100000000
        if service.recent_failed_count > 0:
            backoff_time = service.recent_failed_time + min(120.0, 2**service.recent_failed_count)
            if backoff_time > current_time:
                sadness += 100000000
        modified_num_connections = service.num_connections * 2**service.recent_failed_count
        if modified_num_connections >= self.max_connections_per_service:
            sadness += 100000000
        sadness += modified_num_connections
        sadness += service.lucky_number
        return sadness

    @asynccontextmanager
    async def request(
        self,
        method: str,
        url: str,
        address_hint=None,
        **kwargs
    ) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        request_time = time.time()
        timeout = kwargs.get('timeout', 300)

        while True:
            cur_time = time.time()
            if cur_time > self.next_update_time:
                await self._update_services()
                if not self.services:
                    self.next_update_time = cur_time + 8.0 + random.random() * 4.0
                else:
                    self.next_update_time = cur_time + 20.0 + random.random() * 10.0

            if self.services:
                service_sadnesses = [
                    self._calculate_service_sadness(cur_time, service, address_hint)
                    for service in self.services
                ]
                service_index = min(
                    range(len(service_sadnesses)),
                    key=lambda i: service_sadnesses[i],
                )
                service = self.services[service_index]
                if service_sadnesses[service_index] < 1000000:
                    if address_hint is not None:
                        address_hint[:] = [service.address]
                    break
            await asyncio.sleep(1.0)

        service.num_connections += 1
        statuscode = None
        try:
            url_parts = list(urlparse(url))
            if service.address.endswith("443"):
                url_parts[0] = 'https'
                url_parts[1] = service.address.replace(":443", "")
            else:
                url_parts[0] = 'http'
                url_parts[1] = service.address
            url = urlunparse(url_parts)
            logger.debug(f'Requesting {method} {url} from {service.address}')

            async with aiohttp.ClientSession() as session:
                kwargs['timeout'] = aiohttp.ClientTimeout(total=timeout)
                async with session.request(method, url, **kwargs) as response:
                    statuscode = response.status
                    yield response
        finally:
            response_time = time.time()
            if statuscode is None or statuscode >= 500:
                if response_time - service.recent_failed_time > 1.0:
                    service.recent_failed_count = min(10, service.recent_failed_count + 1)
                    service.recent_failed_time = request_time
            elif statuscode < 400:
                service.recent_failed_count = max(0, service.recent_failed_count - 1)
            service.num_connections -= 1


class AsyncConsulOpenAIProxy:
    def __init__(
        self,
        model: str,
        timeout: float = 600.0,
        max_connections_per_service: int = 256,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.max_connections_per_service = max_connections_per_service
        self.proxy = AsyncConsulLoadBalanceProxy(
            service_name=model,
            max_connections_per_service=max_connections_per_service,
        )

    @property
    def base_url(self) -> str:
        return f'https://{self.model}.app.msh.team/v1'

    async def chat_completions(self, **kwargs):
        if 'model' in kwargs and kwargs['model'] != self.model:
            raise ValueError(f"Model mismatch: {kwargs['model']} != {self.model}")
        kwargs['model'] = self.model

        async with self.proxy.request(
            'POST',
            f'{self.base_url}/chat/completions',
            json=kwargs,
            timeout=self.timeout,
        ) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"HTTP {response.status}: {text}")
            return await response.json()

    async def close(self):
        pass


if __name__ == '__main__':
    service_name = 'gpt-oss-120b-judge'
    timeout = 600.0
    max_connections_per_service = 256
    async def test_llm_client(service_name, timeout, max_connections_per_service):
        llm_client = AsyncConsulOpenAIProxy(
            model=service_name,
            timeout=timeout,
            max_connections_per_service=max_connections_per_service
        )
        messages = [{"role": "user", "content": "谁刺杀了林肯？"}]
        
        result = await llm_client.chat_completions(
            messages=messages,
            max_tokens=4096,
            temperature=0.1
        )
        print(result)

    asyncio.run(test_llm_client(
        service_name=service_name,
        timeout=timeout,
        max_connections_per_service=max_connections_per_service
    ))