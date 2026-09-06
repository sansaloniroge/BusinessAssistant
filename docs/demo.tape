
# Generates docs/demo.gif. Requires the API running locally with real
# providers (see README "How to run it") and the sample docs ingested.
# Run: vhs docs/demo.tape

Output docs/demo.gif

Set Shell "bash"
Set FontSize 17
Set Width 1200
Set Height 520
Set Theme "Dracula"
Set TypingSpeed 8ms
Set Padding 20

Type "# A question the ingested docs actually answer:"
Enter
Sleep 500ms

Type `curl -s localhost:8000/chat -H "X-Tenant-Id: tenant_test" -H "X-User-Id: 11111111-1111-1111-1111-111111111111" -H "Content-Type: application/json" -d '{"message": "¿Con qué frecuencia hay que publicar actualizaciones durante un incidente Sev1 abierto?", "mode": "strict"}' | jq '{answer, cited_from: [.citations[].title]}'`
Enter
Sleep 5s

Type "# A question nothing in the docs covers:"
Enter
Sleep 500ms

Type `curl -s localhost:8000/chat -H "X-Tenant-Id: tenant_test" -H "X-User-Id: 11111111-1111-1111-1111-111111111111" -H "Content-Type: application/json" -d '{"message": "¿Cuál es la política de vacaciones y días festivos?", "mode": "strict"}' | jq '{answer, refused, refusal_reason}'`
Enter
Sleep 5s
