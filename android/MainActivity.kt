package com.example.scamdetect

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Base64
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okio.ByteString
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

private const val TAG = "ScamDetect"
private const val SAMPLE_RATE = 16000
private const val CHUNK_SECONDS = 4
private const val OVERLAP_SECONDS = 0.5f
private const val OVERLAP_SAMPLES = (SAMPLE_RATE * OVERLAP_SECONDS).toInt()  // 8000 samples

// ── Data classes ────────────────────────────────────────────────────────────

data class AnalysisResult(
    val inputType: String,        // "audio" or "document"
    val transcription: String,
    val description: String,
    val verdict: String,          // "SCAM", "LEGITIMATE", "UNCERTAIN"
    val summary: String,
    val recommendations: List<String>,
    val llmTime: Double,
    val callVerdict: String? = null,        // progressive call-level verdict (live mode)
    val segmentVerdict: String? = null,     // per-segment verdict (live mode)
)

data class TestAudioFile(
    val name: String,
    val label: String,
    val category: String,  // "scam", "legit", "suspicious"
)

data class TestImageFile(
    val name: String,
    val label: String,
)

enum class AppTab { LIVE, TEST }

// ── Colors ──────────────────────────────────────────────────────────────────

val ScamRed = Color(0xFFEF4444)
val LegitGreen = Color(0xFF22C55E)
val UncertainYellow = Color(0xFFEAB308)
val DarkBg = Color(0xFF0A0A12)
val SurfaceColor = Color(0xFF1A1A24)
val TextDim = Color(0xFF5A5D66)
val TextLight = Color(0xFFC8CAD0)

fun verdictColor(verdict: String): Color = when (verdict) {
    "SCAM" -> ScamRed
    "LEGITIMATE" -> LegitGreen
    else -> UncertainYellow
}

// ── Main Activity ───────────────────────────────────────────────────────────

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme(
                colorScheme = darkColorScheme(
                    background = DarkBg,
                    surface = SurfaceColor,
                )
            ) {
                ScamDetectApp()
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ScamDetectApp() {
    val context = LocalContext.current
    var serverIp by remember { mutableStateOf("192.168.0.100") }
    var serverPort by remember { mutableStateOf("8001") }
    var isMonitoring by remember { mutableStateOf(false) }
    var isConnected by remember { mutableStateOf(false) }
    var statusText by remember { mutableStateOf("Disconnected") }
    val results = remember { mutableStateListOf<AnalysisResult>() }
    val listState = rememberLazyListState()
    val scope = rememberCoroutineScope()
    var activeTab by remember { mutableStateOf(AppTab.LIVE) }

    // Live streaming state
    var callVerdict by remember { mutableStateOf<String?>(null) }
    var runningTranscript by remember { mutableStateOf("") }

    // Test files state
    val testAudioFiles = remember { mutableStateListOf<TestAudioFile>() }
    val testImageFiles = remember { mutableStateListOf<TestImageFile>() }
    var testFilesLoaded by remember { mutableStateOf(false) }
    var sendingFile by remember { mutableStateOf<String?>(null) }

    // WebSocket + audio state
    var webSocket by remember { mutableStateOf<WebSocket?>(null) }
    var audioJob by remember { mutableStateOf<Job?>(null) }
    val httpClient = remember { OkHttpClient() }

    // Permission launcher
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (!granted) {
            statusText = "Mic permission denied"
        }
    }

    // Image picker for document analysis
    val imagePickerLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let {
            scope.launch(Dispatchers.IO) {
                try {
                    val inputStream = context.contentResolver.openInputStream(it)
                    val bytes = inputStream?.readBytes() ?: return@launch
                    inputStream.close()
                    val b64 = Base64.encodeToString(bytes, Base64.NO_WRAP)
                    val payload = JSONObject().put("image", b64).toString()
                    webSocket?.send(payload)
                    withContext(Dispatchers.Main) { statusText = "Analyzing image..." }
                } catch (e: Exception) {
                    Log.e(TAG, "Image send failed", e)
                }
            }
        }
    }

    // Fetch test files list from server
    fun loadTestFiles() {
        scope.launch(Dispatchers.IO) {
            try {
                val url = "http://$serverIp:$serverPort/api/test-files"
                val request = Request.Builder().url(url).build()
                val response = httpClient.newCall(request).execute()
                val body = response.body?.string() ?: return@launch
                val json = JSONObject(body)

                val audioArr = json.optJSONArray("audio") ?: JSONArray()
                val imageArr = json.optJSONArray("images") ?: JSONArray()

                val audioList = mutableListOf<TestAudioFile>()
                for (i in 0 until audioArr.length()) {
                    val obj = audioArr.getJSONObject(i)
                    audioList.add(TestAudioFile(
                        name = obj.getString("name"),
                        label = obj.getString("label"),
                        category = obj.getString("category"),
                    ))
                }
                val imageList = mutableListOf<TestImageFile>()
                for (i in 0 until imageArr.length()) {
                    val obj = imageArr.getJSONObject(i)
                    imageList.add(TestImageFile(
                        name = obj.getString("name"),
                        label = obj.getString("label"),
                    ))
                }

                withContext(Dispatchers.Main) {
                    testAudioFiles.clear()
                    testAudioFiles.addAll(audioList)
                    testImageFiles.clear()
                    testImageFiles.addAll(imageList)
                    testFilesLoaded = true
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load test files", e)
                withContext(Dispatchers.Main) {
                    statusText = "Failed to load test files: ${e.message}"
                }
            }
        }
    }

    // Clear server history before each test for unbiased results
    fun clearHistory() {
        webSocket?.send(JSONObject().put("clear_history", true).toString())
    }

    // Send test audio file via WebSocket
    fun sendTestAudio(filename: String) {
        if (webSocket == null) return
        sendingFile = filename
        clearHistory()
        scope.launch(Dispatchers.IO) {
            try {
                val url = "http://$serverIp:$serverPort/api/test-audio/$filename.wav"
                val request = Request.Builder().url(url).build()
                val response = httpClient.newCall(request).execute()
                val bytes = response.body?.bytes() ?: return@launch
                val b64 = Base64.encodeToString(bytes, Base64.NO_WRAP)
                val payload = JSONObject().put("audio", b64).toString()
                webSocket?.send(payload)
                withContext(Dispatchers.Main) {
                    statusText = "🟡 Analyzing: ${filename.replace("_", " ")}..."
                }
            } catch (e: Exception) {
                Log.e(TAG, "Send test audio failed", e)
            } finally {
                withContext(Dispatchers.Main) { sendingFile = null }
            }
        }
    }

    // Send test image file via WebSocket
    fun sendTestImage(filename: String) {
        if (webSocket == null) return
        sendingFile = filename
        clearHistory()
        scope.launch(Dispatchers.IO) {
            try {
                val url = "http://$serverIp:$serverPort/api/test-image/$filename"
                val request = Request.Builder().url(url).build()
                val response = httpClient.newCall(request).execute()
                val bytes = response.body?.bytes() ?: return@launch
                val b64 = Base64.encodeToString(bytes, Base64.NO_WRAP)
                val payload = JSONObject().put("image", b64).toString()
                webSocket?.send(payload)
                withContext(Dispatchers.Main) {
                    statusText = "🟡 Analyzing image..."
                }
            } catch (e: Exception) {
                Log.e(TAG, "Send test image failed", e)
            } finally {
                withContext(Dispatchers.Main) { sendingFile = null }
            }
        }
    }

    // WebSocket message handler
    fun handleMessage(text: String) {
        try {
            val json = JSONObject(text)
            if (json.optString("type") != "result") return

            val recs = mutableListOf<String>()
            val recsArray = json.optJSONArray("recommendations")
            if (recsArray != null) {
                for (i in 0 until recsArray.length()) {
                    recs.add(recsArray.getString(i))
                }
            }

            val result = AnalysisResult(
                inputType = json.optString("input_type", "audio"),
                transcription = json.optString("transcription", ""),
                description = json.optString("description", ""),
                verdict = json.optString("verdict", "UNCERTAIN"),
                summary = json.optString("summary", ""),
                recommendations = recs,
                llmTime = json.optDouble("llm_time", 0.0),
                callVerdict = if (json.has("call_verdict")) json.optString("call_verdict") else null,
                segmentVerdict = if (json.has("segment_verdict")) json.optString("segment_verdict") else null,
            )
            results.add(result)
            scope.launch { listState.animateScrollToItem(results.size - 1) }

            // Update live streaming state
            if (result.callVerdict != null) {
                callVerdict = result.callVerdict
            }
            val transcript = json.optString("running_transcript", "")
            if (transcript.isNotBlank()) {
                runningTranscript = transcript
            }

            // Update status based on verdict (use call verdict in live mode)
            val displayVerdict = result.callVerdict ?: result.verdict
            statusText = when (displayVerdict) {
                "SCAM" -> "⚠️ SCAM DETECTED"
                "LEGITIMATE" -> "✅ Legitimate"
                else -> "⚡ Uncertain"
            }
        } catch (e: Exception) {
            Log.e(TAG, "Parse error", e)
        }
    }

    // Connect WebSocket
    fun connect() {
        val client = OkHttpClient()
        val mode = if (activeTab == AppTab.LIVE) "live" else "eval"
        val url = "ws://$serverIp:$serverPort/ws?mode=$mode"
        val request = Request.Builder().url(url).build()
        statusText = "Connecting to $url..."

        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, response: Response) {
                scope.launch(Dispatchers.Main) {
                    isConnected = true
                    statusText = "Connected — listening"
                }
            }

            override fun onMessage(ws: WebSocket, text: String) {
                scope.launch(Dispatchers.Main) { handleMessage(text) }
            }

            override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
                scope.launch(Dispatchers.Main) {
                    isConnected = false
                    isMonitoring = false
                    statusText = "Connection failed: ${t.message}"
                }
            }

            override fun onClosed(ws: WebSocket, code: Int, reason: String) {
                scope.launch(Dispatchers.Main) {
                    isConnected = false
                    isMonitoring = false
                    statusText = "Disconnected"
                }
            }
        })
    }

    // Start audio capture + send loop
    fun startMonitoring() {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
            return
        }

        connect()

        audioJob = scope.launch(Dispatchers.IO) {
            // Wait for connection
            delay(2000)
            if (!isConnected) return@launch

            val bufferSize = maxOf(
                AudioRecord.getMinBufferSize(
                    SAMPLE_RATE,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT
                ),
                SAMPLE_RATE * 2 * CHUNK_SECONDS
            )

            val recorder = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize
            )

            if (recorder.state != AudioRecord.STATE_INITIALIZED) {
                withContext(Dispatchers.Main) { statusText = "AudioRecord init failed" }
                return@launch
            }

            recorder.startRecording()
            withContext(Dispatchers.Main) {
                isMonitoring = true
                statusText = "🟢 Monitoring — speak on speakerphone"
            }

            val chunkSamples = SAMPLE_RATE * CHUNK_SECONDS
            val chunkBuffer = ShortArray(chunkSamples)
            val overlapBuffer = ShortArray(OVERLAP_SAMPLES)
            var hasOverlap = false

            try {
                while (isActive) {
                    // Prepend overlap from previous chunk
                    val startOffset = if (hasOverlap) OVERLAP_SAMPLES else 0
                    if (hasOverlap) {
                        System.arraycopy(overlapBuffer, 0, chunkBuffer, 0, OVERLAP_SAMPLES)
                    }

                    // Read new audio to fill the rest of the chunk
                    var offset = startOffset
                    while (offset < chunkSamples && isActive) {
                        val read = recorder.read(chunkBuffer, offset, chunkSamples - offset)
                        if (read > 0) offset += read
                        else break
                    }

                    if (!isActive || offset == 0) break

                    // Save last 0.5s for overlap with next chunk
                    if (offset >= OVERLAP_SAMPLES) {
                        System.arraycopy(chunkBuffer, offset - OVERLAP_SAMPLES, overlapBuffer, 0, OVERLAP_SAMPLES)
                        hasOverlap = true
                    }

                    // Check if there's actual audio (RMS energy gate)
                    val rms = Math.sqrt(chunkBuffer.take(offset).sumOf { it.toLong() * it.toLong() }.toDouble() / offset)
                    if (rms < 50) continue  // skip near-silence

                    // Encode as WAV
                    val wav = encodeWav(chunkBuffer, offset, SAMPLE_RATE)
                    val b64 = Base64.encodeToString(wav, Base64.NO_WRAP)
                    val payload = JSONObject().put("audio", b64).toString()

                    webSocket?.send(payload)
                    withContext(Dispatchers.Main) { statusText = "🟡 Analyzing..." }
                }
            } finally {
                recorder.stop()
                recorder.release()
            }
        }
    }

    fun stopMonitoring() {
        audioJob?.cancel()
        audioJob = null
        webSocket?.close(1000, "User stopped")
        webSocket = null
        isMonitoring = false
        isConnected = false
        statusText = "Stopped"
        callVerdict = null
        runningTranscript = ""
    }

    // ── UI ──────────────────────────────────────────────────────────────────

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Scam Detect", fontWeight = FontWeight.Bold) },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = DarkBg,
                    titleContentColor = Color.White,
                )
            )
        },
        containerColor = DarkBg,
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            // Server config
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(
                    value = serverIp,
                    onValueChange = { serverIp = it },
                    label = { Text("Server IP") },
                    modifier = Modifier.weight(2f),
                    singleLine = true,
                    enabled = !isMonitoring,
                )
                OutlinedTextField(
                    value = serverPort,
                    onValueChange = { serverPort = it },
                    label = { Text("Port") },
                    modifier = Modifier.weight(1f),
                    singleLine = true,
                    enabled = !isMonitoring,
                )
            }

            // Status bar
            Surface(
                shape = RoundedCornerShape(12.dp),
                color = SurfaceColor,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(
                    text = statusText,
                    modifier = Modifier.padding(12.dp),
                    color = when {
                        statusText.contains("SCAM") -> ScamRed
                        statusText.contains("Monitoring") || statusText.contains("Legitimate") -> LegitGreen
                        statusText.contains("Analyzing") || statusText.contains("Uncertain") -> UncertainYellow
                        else -> TextDim
                    },
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Medium,
                )
            }

            // Tab row
            TabRow(
                selectedTabIndex = if (activeTab == AppTab.LIVE) 0 else 1,
                containerColor = SurfaceColor,
                contentColor = Color.White,
            ) {
                Tab(
                    selected = activeTab == AppTab.LIVE,
                    onClick = { activeTab = AppTab.LIVE },
                    text = { Text("🎙 Live") },
                )
                Tab(
                    selected = activeTab == AppTab.TEST,
                    onClick = {
                        activeTab = AppTab.TEST
                        if (!testFilesLoaded) loadTestFiles()
                    },
                    text = { Text("🧪 Test Files") },
                )
            }

            if (activeTab == AppTab.LIVE) {
                // Call verdict banner (live mode — shows progressive verdict)
                if (callVerdict != null && isMonitoring) {
                    Surface(
                        shape = RoundedCornerShape(12.dp),
                        color = verdictColor(callVerdict!!).copy(alpha = 0.15f),
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Column(modifier = Modifier.padding(12.dp)) {
                            Text(
                                text = "Call Verdict: ${callVerdict}",
                                color = verdictColor(callVerdict!!),
                                fontSize = 16.sp,
                                fontWeight = FontWeight.Bold,
                            )
                        }
                    }
                }

                // Live transcript (scrolling)
                if (runningTranscript.isNotBlank() && isMonitoring) {
                    Surface(
                        shape = RoundedCornerShape(8.dp),
                        color = SurfaceColor,
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(max = 120.dp),
                    ) {
                        val scrollState = rememberScrollState()
                        LaunchedEffect(runningTranscript) {
                            scrollState.animateScrollTo(scrollState.maxValue)
                        }
                        Column(
                            modifier = Modifier
                                .padding(10.dp)
                                .verticalScroll(scrollState),
                        ) {
                            Text(
                                text = runningTranscript,
                                color = TextDim,
                                fontSize = 12.sp,
                                lineHeight = 18.sp,
                            )
                        }
                    }
                }

                // Results list
                LazyColumn(
                    state = listState,
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    items(results) { result ->
                        ResultCard(result)
                    }
                }

                // Controls
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Button(
                        onClick = { if (isMonitoring) stopMonitoring() else startMonitoring() },
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (isMonitoring) ScamRed else LegitGreen
                        ),
                    ) {
                        Text(if (isMonitoring) "Stop" else "Start Monitoring")
                    }

                    OutlinedButton(
                        onClick = { imagePickerLauncher.launch("image/*") },
                        modifier = Modifier.weight(1f),
                        enabled = isConnected,
                    ) {
                        Text("📄 Scan Document")
                    }
                }

                // Clear button
                TextButton(
                    onClick = { results.clear() },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text("Clear History", color = TextDim)
                }
            } else {
                // TEST TAB
                TestFilesPanel(
                    audioFiles = testAudioFiles,
                    imageFiles = testImageFiles,
                    isLoaded = testFilesLoaded,
                    isConnected = isConnected,
                    sendingFile = sendingFile,
                    results = results,
                    listState = listState,
                    onSendAudio = { sendTestAudio(it) },
                    onSendImage = { sendTestImage(it) },
                    onConnect = {
                        if (!isConnected) {
                            connect()
                            scope.launch {
                                delay(2000)
                                if (isConnected && !testFilesLoaded) loadTestFiles()
                            }
                        }
                    },
                    onReload = { loadTestFiles() },
                )
            }

            Spacer(modifier = Modifier.height(8.dp))
        }
    }
}

// ── Test Files Panel ────────────────────────────────────────────────────────

@Composable
fun TestFilesPanel(
    audioFiles: List<TestAudioFile>,
    imageFiles: List<TestImageFile>,
    isLoaded: Boolean,
    isConnected: Boolean,
    sendingFile: String?,
    results: List<AnalysisResult>,
    listState: androidx.compose.foundation.lazy.LazyListState,
    onSendAudio: (String) -> Unit,
    onSendImage: (String) -> Unit,
    onConnect: () -> Unit,
    onReload: () -> Unit,
) {
    if (!isConnected) {
        Button(
            onClick = onConnect,
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(containerColor = LegitGreen),
        ) {
            Text("Connect to Server")
        }
        return
    }

    if (!isLoaded) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.Center,
        ) {
            CircularProgressIndicator(color = UncertainYellow, modifier = Modifier.size(24.dp))
            Spacer(modifier = Modifier.width(8.dp))
            Text("Loading test files...", color = TextDim)
        }
        return
    }

    LazyColumn(
        state = listState,
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        // ── Scam audio ──
        val scamFiles = audioFiles.filter { it.category == "scam" }
        if (scamFiles.isNotEmpty()) {
            item {
                Text("🔴 Scam Audio", color = ScamRed, fontWeight = FontWeight.Bold,
                    fontSize = 13.sp, modifier = Modifier.padding(top = 4.dp))
            }
            items(scamFiles) { file ->
                TestFileChip(
                    label = file.label,
                    icon = "🎙",
                    color = ScamRed,
                    isSending = sendingFile == file.name,
                    onClick = { onSendAudio(file.name) },
                )
            }
        }

        // ── Suspicious audio ──
        val suspiciousFiles = audioFiles.filter { it.category == "suspicious" }
        if (suspiciousFiles.isNotEmpty()) {
            item {
                Text("🟡 Suspicious Audio", color = UncertainYellow, fontWeight = FontWeight.Bold,
                    fontSize = 13.sp, modifier = Modifier.padding(top = 8.dp))
            }
            items(suspiciousFiles) { file ->
                TestFileChip(
                    label = file.label,
                    icon = "🎙",
                    color = UncertainYellow,
                    isSending = sendingFile == file.name,
                    onClick = { onSendAudio(file.name) },
                )
            }
        }

        // ── Legit audio ──
        val legitFiles = audioFiles.filter { it.category == "legit" }
        if (legitFiles.isNotEmpty()) {
            item {
                Text("🟢 Legitimate Audio", color = LegitGreen, fontWeight = FontWeight.Bold,
                    fontSize = 13.sp, modifier = Modifier.padding(top = 8.dp))
            }
            items(legitFiles) { file ->
                TestFileChip(
                    label = file.label,
                    icon = "🎙",
                    color = LegitGreen,
                    isSending = sendingFile == file.name,
                    onClick = { onSendAudio(file.name) },
                )
            }
        }

        // ── Test images ──
        if (imageFiles.isNotEmpty()) {
            item {
                Text("📄 Test Images", color = TextLight, fontWeight = FontWeight.Bold,
                    fontSize = 13.sp, modifier = Modifier.padding(top = 8.dp))
            }
            items(imageFiles) { file ->
                TestFileChip(
                    label = file.label,
                    icon = "📄",
                    color = UncertainYellow,
                    isSending = sendingFile == file.name,
                    onClick = { onSendImage(file.name) },
                )
            }
        }

        // ── Results from test runs ──
        if (results.isNotEmpty()) {
            item {
                Text("Results", color = TextLight, fontWeight = FontWeight.Bold,
                    fontSize = 13.sp, modifier = Modifier.padding(top = 12.dp))
            }
            items(results) { result ->
                ResultCard(result)
            }
        }

        // Reload button
        item {
            TextButton(onClick = onReload, modifier = Modifier.fillMaxWidth()) {
                Text("Reload Test Files", color = TextDim)
            }
        }
    }
}

@Composable
fun TestFileChip(
    label: String,
    icon: String,
    color: Color,
    isSending: Boolean,
    onClick: () -> Unit,
) {
    Surface(
        shape = RoundedCornerShape(8.dp),
        color = color.copy(alpha = 0.1f),
        modifier = Modifier
            .fillMaxWidth()
            .clickable(enabled = !isSending) { onClick() },
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            if (isSending) {
                CircularProgressIndicator(
                    color = color, modifier = Modifier.size(16.dp), strokeWidth = 2.dp
                )
            } else {
                Text(icon, fontSize = 14.sp)
            }
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = label,
                color = TextLight,
                fontSize = 13.sp,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.weight(1f),
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text("Send ▶", color = color, fontSize = 11.sp, fontWeight = FontWeight.Bold)
        }
    }
}

// ── Result Card ─────────────────────────────────────────────────────────────

@Composable
fun ResultCard(result: AnalysisResult) {
    Surface(
        shape = RoundedCornerShape(12.dp),
        color = SurfaceColor,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            // Header (transcription or description)
            val icon = if (result.inputType == "document") "📄" else "🎙"
            val headerText = if (result.inputType == "document")
                result.description else result.transcription

            if (headerText.isNotBlank()) {
                Text(
                    text = "$icon $headerText",
                    color = TextLight,
                    fontSize = 13.sp,
                    lineHeight = 20.sp,
                )
                Spacer(modifier = Modifier.height(8.dp))
            }

            // Verdict badge
            Surface(
                shape = RoundedCornerShape(6.dp),
                color = verdictColor(result.verdict).copy(alpha = 0.15f),
            ) {
                Text(
                    text = result.verdict,
                    color = verdictColor(result.verdict),
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 3.dp),
                )
            }

            // Summary
            if (result.summary.isNotBlank()) {
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = result.summary,
                    color = TextDim,
                    fontSize = 12.sp,
                    lineHeight = 18.sp,
                )
            }

            // Recommendations
            if (result.recommendations.isNotEmpty()) {
                Spacer(modifier = Modifier.height(8.dp))
                result.recommendations.forEach { rec ->
                    Text(
                        text = "→ $rec",
                        color = TextLight,
                        fontSize = 12.sp,
                        lineHeight = 18.sp,
                        modifier = Modifier.padding(start = 4.dp, bottom = 2.dp),
                    )
                }
            }

            // Meta
            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = "LLM ${result.llmTime}s",
                color = TextDim,
                fontSize = 10.sp,
            )
        }
    }
}

// ── WAV Encoder ─────────────────────────────────────────────────────────────

fun encodeWav(samples: ShortArray, count: Int, sampleRate: Int): ByteArray {
    val dataSize = count * 2
    val out = ByteArrayOutputStream(44 + dataSize)
    val dos = DataOutputStream(out)

    // RIFF header
    dos.writeBytes("RIFF")
    dos.writeInt(Integer.reverseBytes(36 + dataSize))
    dos.writeBytes("WAVE")

    // fmt chunk
    dos.writeBytes("fmt ")
    dos.writeInt(Integer.reverseBytes(16))       // chunk size
    dos.writeShort(java.lang.Short.reverseBytes(1).toInt())   // PCM
    dos.writeShort(java.lang.Short.reverseBytes(1).toInt())   // mono
    dos.writeInt(Integer.reverseBytes(sampleRate))
    dos.writeInt(Integer.reverseBytes(sampleRate * 2))        // byte rate
    dos.writeShort(java.lang.Short.reverseBytes(2).toInt())   // block align
    dos.writeShort(java.lang.Short.reverseBytes(16).toInt())  // bits per sample

    // data chunk
    dos.writeBytes("data")
    dos.writeInt(Integer.reverseBytes(dataSize))

    val buf = ByteBuffer.allocate(count * 2).order(ByteOrder.LITTLE_ENDIAN)
    for (i in 0 until count) {
        buf.putShort(samples[i])
    }
    out.write(buf.array())

    return out.toByteArray()
}
