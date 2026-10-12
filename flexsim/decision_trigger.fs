/** Custom Code - HAAM-RL decision trigger
 *
 * One call = one decision of the agent. The Python API decides the phase and the action:
 *   phase "fill"   -> only store the waiting vehicle in input_buffer
 *   phase "steady" -> paint the front of output_buffer, THEN store the waiting vehicle in input_buffer
 *   phase "drain"  -> only paint the front of output_buffer
 * FlexSim just executes what the API answers, so the twin and FlexSim follow the same rules.
 *
 * Objects expected in the model: Waiting_Queue (or Queue1), Painting_Machine (or Processor1),
 * Buffer_Queue1..Buffer_Queue5, and a global table KPI_Model with columns ColorActual, CambiosColor.
 * Vehicles carry their color (1..6) in the label "Type".
 */

int LOOKAHEAD = 10;          // vehicles behind the waiting one sent to the API (extra ones are ignored)
string API_HOST = "127.0.0.1";
int API_PORT = 8000;

// ------------------------------------------------------------------ 1. Objects
Object waitingQueue = Model.find("Waiting_Queue");
if (!waitingQueue) waitingQueue = Model.find("Queue1");
Object machine = Model.find("Painting_Machine");
if (!machine) machine = Model.find("Processor1");
if (!waitingQueue || !machine) {
    print("ERROR: Waiting_Queue or Painting_Machine not found.");
    return 0;
}

Array buffers = Array(5);
int stored = 0;
for (int b = 1; b <= 5; b++) {
    buffers[b] = Model.find("Buffer_Queue" + string.fromNum(b));
    if (!buffers[b]) {
        print("ERROR: Buffer_Queue", b, " not found.");
        return 0;
    }
    stored = stored + content(buffers[b]);
}

// ------------------------------------------------------------------ 2. KPI table (reset at model start)
Table kpi = Table("KPI_Model");
if (time() < 4) {
    kpi[1]["ColorActual"] = 0;
    kpi[1]["CambiosColor"] = 0;
}
int lastPainted = kpi[1]["ColorActual"];
int colorChanges = kpi[1]["CambiosColor"];

if (content(waitingQueue) == 0 && stored == 0) {
    print("End of episode: no vehicles left. Color changes: ", colorChanges);
    stop();
    return 0;
}

// ------------------------------------------------------------------ 3. State: incoming, last painted, 5 x 5 slots
int incoming = 0;
if (content(waitingQueue) > 0) incoming = getlabel(rank(waitingQueue, 1), "Type");

string state = "[" + string.fromNum(incoming) + "," + string.fromNum(lastPainted);
for (int b = 1; b <= 5; b++) {
    for (int slot = 1; slot <= 5; slot++) {
        int color = 0;
        if (content(buffers[b]) >= slot) color = getlabel(rank(buffers[b], slot), "Type");
        state = state + "," + string.fromNum(color);
    }
}
state = state + "]";

string upcoming = "[";
int queued = content(waitingQueue);
for (int k = 2; k <= queued && k <= LOOKAHEAD + 1; k++) {
    if (k > 2) upcoming = upcoming + ",";
    upcoming = upcoming + string.fromNum(getlabel(rank(waitingQueue, k), "Type"));
}
upcoming = upcoming + "]";

// ------------------------------------------------------------------ 4. Ask the API
Http.Request request;
request.host = API_HOST;
request.port = API_PORT;
request.path = "/step";
request.method = Http.Method.Post;
request.headers = "Content-Type: application/json\r\n";
request.timeout = 5000;
request.data = "{\"state_vector\":" + state + ",\"upcoming\":" + upcoming + "}";

Http.Response response = request.sendAndWait();
if (response.statusCode != 200) {
    print("ERROR: API answered HTTP ", response.statusCode, ": ", response.value);
    return 0;
}

Map answer = JSON.parse(response.value);
int actionId = answer["action_id"];
int inputNumber = answer["input_buffer"];    // 1-based
int outputNumber = answer["output_buffer"];  // 1-based
string phase = answer["phase"];
int extract = (phase == "steady" || phase == "drain");
int insert = (phase == "fill" || phase == "steady");

Object inputBuffer = buffers[inputNumber];
Object outputBuffer = buffers[outputNumber];

// ------------------------------------------------------------------ 5. Validate before touching anything
if (extract && content(outputBuffer) == 0) {
    print("ERROR: action ", actionId, " extracts from empty Buffer_Queue", outputNumber);
    return 0;
}
int freedSlot = extract && outputBuffer == inputBuffer;
if (insert && content(inputBuffer) >= 5 && !freedSlot) {
    print("ERROR: action ", actionId, " stores into full Buffer_Queue", inputNumber);
    return 0;
}

// ------------------------------------------------------------------ 6. Extract first: paint the front vehicle
int painted = 0;
if (extract) {
    treenode vehicle = first(outputBuffer);
    painted = getlabel(vehicle, "Type");
    machine.input.open();
    moveobject(vehicle, machine);
    // The KPI is updated only after the vehicle really moved to the machine.
    if (lastPainted > 0 && painted != lastPainted) {
        colorChanges = colorChanges + 1;
        kpi[1]["CambiosColor"] = colorChanges;
    }
    kpi[1]["ColorActual"] = painted;
}

// ------------------------------------------------------------------ 7. Then store the waiting vehicle
if (insert) {
    moveobject(rank(waitingQueue, 1), inputBuffer);
}

print("HAAM-RL [", phase, "] action ", actionId, ": in B", inputNumber, " out B", outputNumber,
      " | painted ", painted, " | color changes ", colorChanges, " | decided by ", answer["decided_by"]);
return 0;
