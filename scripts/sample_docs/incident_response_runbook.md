---
title: Runbook de respuesta a incidentes de producción
department: IT
doc_type: runbook
tags: [incidentes, produccion, on-call]
doc_date: 2026-03-01
---

# Runbook de respuesta a incidentes de producción

Este runbook describe los pasos a seguir cuando salta una alerta de producción fuera del horario laboral habitual.

## Clasificación de severidad

Sev1 implica caída total o parcial de un servicio con impacto directo en clientes (por ejemplo, la API de chat no responde). Sev2 implica degradación de rendimiento visible pero sin caída total. Sev3 son alertas informativas que no requieren acción inmediata. Solo Sev1 y Sev2 activan el protocolo de guardia fuera de horario.

## Primeros cinco minutos

La persona de guardia debe confirmar la alerta en el dashboard de observabilidad antes de escalar, para descartar falsos positivos. Si se confirma, debe publicar un mensaje en el canal #incidentes con severidad, servicio afectado y hora de inicio, incluso antes de tener diagnóstico, para que el resto del equipo tenga visibilidad temprana.

## Escalado

Si en 15 minutos no se ha identificado la causa raíz, la persona de guardia debe escalar a un segundo ingeniero según la lista de guardia secundaria, sin esperar a agotar el tiempo por orgullo o por querer resolverlo en solitario. Para Sev1 que supere los 30 minutos sin resolución, se notifica automáticamente al responsable de ingeniería.

## Comunicación durante el incidente

Se debe publicar una actualización en el canal #incidentes cada 20 minutos mientras el incidente esté abierto, aunque no haya novedades sustanciales ("seguimos investigando" es una actualización válida). El silencio prolongado genera más ansiedad en el resto del equipo que una actualización sin noticias nuevas.

## Cierre y post-mortem

Todo incidente Sev1 requiere un post-mortem escrito en un plazo de 5 días laborables, con causa raíz, cronología y acciones de seguimiento con responsable y fecha. El post-mortem es explícitamente no punitivo: el objetivo es identificar fallos del sistema, no de las personas.
