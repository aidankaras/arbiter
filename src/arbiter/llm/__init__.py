"""Provider routing, cost metering, prompt caching, and budget enforcement.

Every model call in the system passes through here so that cost is attributable
to a packet and a prediction, and so that token and spend ceilings are enforced
in code rather than requested in a prompt.
"""
