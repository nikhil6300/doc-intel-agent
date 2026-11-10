# Production Incident Response Runbook

Incidents are classified by severity. Severity 1 means a full outage or confirmed data exposure. Severity 2 means major degradation affecting more than 20 percent of users. Severity 3 means a minor issue with a workaround.

When a Severity 1 incident is declared, the on-call engineer becomes the incident commander and opens a dedicated communication channel. The commander assigns a communications lead who posts status updates every 30 minutes until the incident is resolved.

For Severity 1 incidents the first priority is to restore service, using rollback to the last known good release if the cause is not immediately clear. Debugging the root cause happens after service is restored, not before.

Within five working days of resolving any Severity 1 or Severity 2 incident the team must publish a blameless post-incident review. The review includes a timeline, the root cause, what went well, what went poorly, and action items with named owners and due dates.

Security-related incidents must additionally be reported to the information security team within one hour of detection.
