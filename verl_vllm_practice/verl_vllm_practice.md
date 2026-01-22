<style>
table {
    width: 100%;
    table-layout: fixed;
}
th:nth-child(1), td:nth-child(1) {
    width: 25%;
}
th:nth-child(2), td:nth-child(2) {
    width: 25%;
}
th:nth-child(3), td:nth-child(3) {
    width: 25%;
}
th:nth-child(4), td:nth-child(4) {
    width: 25%;
}
</style>
## verl/vllm 版本问题汇总
| verl/vllm | 0.10.2 | 0.11.0（精度有问题） | 0.11.2（async server 问题，已解决） |
|---|---|---|---|
| 0.5.0 <br> 必碰到init device问题（已解决）  | retool: 【error】 executing method 'init device'  <br> * init device 问题可以通过两个patch来来解决，详见下方| * retool: 【error】ModuleNotFoundError: No module named 'vllm.model_executor.sampling_metadata' 修复后同样 'init device' error <br> * init device 问题可以通过两个patch来来解决，详见下方| -  |
|0.5.0-484-g2c062202（镜像默认） <br> retool 接口有问题，修复后没问题   | retool: compute_score() got an unexpected keyword argument 'reward_router_address'<br>修复后可以正常运行   | retool: compute_score() got an unexpected keyword argument 'reward_router_address'<br>修复后可以正常运行   |  retool 【直接运行会碰到model-hosting-container-standards  from vllm.utils import FlexibleArgumentParser, get_tcp_uri， init_app take 3 position问题，修复后，同样 async server error<br>https://github.com/volcengine/verl/issues/4308#issuecomment-3594293427 可以解 |
| 0.6.0  | verl retool: 【error】ExternalZeroMQDistributedExecutor.collective_rpc() got an unexpected keyword argument 'non_block' <已解决>  |   |   |
| 0.7.0  | retool: TypeError: cannot pickle '_contextvars.ContextVar' object <br> 未解决  |  retool: TypeError: cannot pickle '_contextvars.ContextVar' object <br> 未解决 |   |
|   |   |   |   |


## init device 问题


修改ray和vllm  <br> * 在verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py +476禁用monkey patch可以解决compute_logits问题。 <br>  * 通过之前issue async server error <br>  https://github.com/volcengine/verl/issues/4308#issuecomment-3594293427 可以解决collective_rpc重载non_block问题


## compute_score() 问题
