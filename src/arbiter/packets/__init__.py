"""Evidence packets: the frozen, content-addressed input every arm reads.

The comparison between arms is only meaningful if each saw the same thing. That
is enforced here rather than asserted: an arm is handed a packet, the packet is
immutable, and its identity is the hash of its contents — so two arms that
report the same packet hash provably read the same bytes.
"""
