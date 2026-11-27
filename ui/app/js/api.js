// Acceso a la API. Los errores llevan el mensaje que devuelve el servidor (por ejemplo, qué comando ejecutar).

export async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) {
    let detail = '';
    try { detail = (await res.json()).detail ?? ''; } catch { /* sin cuerpo JSON */ }
    const err = new Error(detail || `Error ${res.status} al pedir ${url}`);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

/** Como getJSON, pero devuelve null si el servidor responde 404 (p. ej. una referencia como SPY no tiene eventos). */
export async function getJSONorNull(url) {
  try {
    return await getJSON(url);
  } catch (err) {
    if (err.status === 404) return null;
    throw err;
  }
}
