# Security scope

This is a loopback-only experimental service for trusted local users, ONNX models and finite numeric inputs. It has no authentication, authorization, tenant isolation, model sandbox, TLS or internet-facing abuse protection. Any local process with access can inspect diagnostics and request cancellation. Do not expose it through a public proxy without designing those protections.

Request bodies are bounded to 8 KiB, tensor values/shapes are validated, and admission/audit retention are bounded. These bounds do not cover every transport connection, kernel buffer or hostile runtime allocation. Active cancellation/deadline response does not interrupt native inference. A failed worker fails closed; the application does not retry a model silently.

Keep credentials, private inputs, personal instructions, local environments and raw runtime logs out of Git. Reports retain public fixture data and host CPU/OS/runtime provenance, without collecting hostname, username or serial identifiers. The publication scanner is heuristic and does not guarantee detection of every secret.

Only trusted fixed-SHA official actions run in CI; workflow tokens are read-only and checkout credentials are not persisted. Do not approve external-code workflows without examining the changes. A green CI run is not a security certification.
