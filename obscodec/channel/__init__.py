from .impairments import (
    AWGNChannel, RayleighFadingChannel, PacketLossChannel,
    BurstPacketLossChannel, HeterogeneousChannel, CompositeChannel,
    evaluate_channel_sweep, evaluate_channel_robustness, get_agent_boundaries,
)
from .adaptive import (
    RateAllocator, AdaptiveDigitalCodec, AgentChannelState,
    compute_coordination_score,
)
