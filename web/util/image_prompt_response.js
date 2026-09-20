export async function readEditResponse(response) {
    const text = await response.text();
    let data;
    try { data = JSON.parse(text); } catch { /* HTTP errors may be plain text or HTML. */ }
    if (!response.ok) {
        if (response.status === 404 || response.status === 405) {
            throw new Error(`Edit API unavailable (HTTP ${response.status}). Restart the ComfyUI server, then reload this page. Your current prompt has not been changed.`);
        }
        throw new Error(data?.error || `Prompt editing failed (HTTP ${response.status}). Your current prompt has not been changed. Check the ComfyUI server log.`);
    }
    if (typeof data?.prompt !== "string" || !data.prompt.trim()) {
        throw new Error("Invalid edit response: the server did not return a prompt. Your current prompt has not been changed.");
    }
    return data;
}
