#include <iostream>
#include <queue>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <utility>
#include <algorithm>
#include <locale>   
#include <boost/functional/hash.hpp>
#include <omp.h>
#include <cstring>
#include <thread>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_hash_map.h>

#define regularTile 1
#define targetTile  2
#define fragileTile 3
#define wall        4
#define maxMap      256
#define maxMapLen   16
#define top         0
#define down        1
#define left        2
#define right       3

using namespace std;
typedef uint8_t ui;
typedef pair<ui, ui> Pair;

// global variable
ui gameMap[maxMap];
unordered_set<ui, boost::hash<ui>> targetPos;
unordered_set<ui, boost::hash<ui>> startPos;
ui endPlayerPos;
ui mapWidth, mapHeight;

// ---- for acceleration ----

// top, down, left, right
// {row/col number of target}
Pair boundary[4];

ui pairToUi(ui x, ui y, ui col) {
    return x * col + y;
}

Pair uiToPair(ui u, ui col) {
    return make_pair(u / col, u % col);
}

struct State {
    unordered_set<ui, boost::hash<ui>> boxes;
    vector<Pair> seq;

    State(unordered_set<ui, boost::hash<ui>> b, vector<Pair> s) :boxes(b), seq(s) {} 
    State() {}

    bool operator==(const State& other) const{
        if (seq.size() > 0 && other.seq.size() > 0) {
            if (seq.back() != other.seq.back()) return false;
        }
        else if (seq.size() == 0 && other.seq.size() == 0) {
            return true;
        }
        else return false;

        int res = 0;
        for (auto s : boxes) res ^= s;
        for (auto s : other.boxes) res ^= s;

        if (res == 0) return true;
        return false;
    };
};

struct StateHash
{
    int operator()( const State& st ) const
    {
        size_t seed = 0;
        for (auto s : st.boxes) boost::hash_combine(seed, s);
        if (st.seq.size() != 0) {
            boost::hash_combine(seed, st.seq.back().first);
            boost::hash_combine(seed, st.seq.back().second);
        }
        return seed;
    }
};

void readMap(char *fileName, unordered_set<ui, boost::hash<ui>> &boxPos) {
    ifstream fin(fileName);
    string line;
    int w, h;
    w = h = 0;
    mapWidth = mapHeight = 0;

    while (getline(fin, line)) {
        for (auto symbol : line) {
            ui coord = pairToUi(w, h, mapHeight);
            h++;
            // The player stepping on a regular tile.
            if (symbol == 'o') {
                endPlayerPos = coord;
                gameMap[coord] = regularTile;
            }
            // The player stepping on a target tile.
            else if (symbol == 'O') {
                endPlayerPos = coord;
                boxPos.insert(coord);
                startPos.insert(coord);
                gameMap[coord] = regularTile;
            }
            // A box on a regular tile.
            else if (symbol == 'x') {
                targetPos.insert(coord);
                gameMap[coord] = targetTile;
            }
            // A box on a target tile.
            else if (symbol == 'X') {
                boxPos.insert(coord);
                startPos.insert(coord);
                targetPos.insert(coord);
                gameMap[coord] = targetTile;
            }
            // (space): Nothing on a regular tile.
            else if (symbol == ' ') {
                gameMap[coord] = regularTile;
            }
            // Nothing on a target tile.
            else if (symbol == '.') {
                boxPos.insert(coord);
                gameMap[coord] = regularTile;
            }
            // Wall.
            else if (symbol == '#') {
                gameMap[coord] = wall;
            }
            // A fragile tile where only the player can step on. The boxes are not allowed to be put on it.
            else if (symbol == '@') {
                gameMap[coord] = fragileTile;
            }
            // The player stepping on a fragile tile.
            else if (symbol == '!') {
                endPlayerPos = coord;
                gameMap[coord] = fragileTile;
            }
        }
        w++;
        mapHeight = h;
        h = 0;
    }
    mapWidth = w;
}

void printMap(State st) {
    if (st.seq.size() == 0) {
        for (int i=0; i<mapWidth; i++) {
            for (int j=0; j<mapHeight; j++) {
                ui pr = pairToUi(i, j, mapHeight);
                if (gameMap[pr] == wall) cout << "#";
                else if (gameMap[pr] == fragileTile) cout << "@";
                else if (st.boxes.find(pr) != st.boxes.end()) {
                    cout << ".";
                }
                else cout << " ";
            }
            cout << endl;
        }
        cout << endl;
        return ;
    }

    ui from, to;
    tie(from, to) = st.seq.back();
    ui dst = to + (to - from);

    for (int i=0; i<mapWidth; i++) {
        for (int j=0; j<mapHeight; j++) {
            ui pr = pairToUi(i, j, mapHeight);
            if (gameMap[pr] == wall) cout << "#";
            else if (gameMap[pr] == fragileTile) cout << "@";
            else if (st.boxes.find(pr) != st.boxes.end()) {
                cout << ".";
            }
            else if (pr == dst) {
                cout << "o";
            }
            else cout << " ";
        }
        cout << endl;
    }
    cout << "\n";
}

void printUi(ui val) {
    cout << (int)uiToPair(val, mapHeight).first << " " << (int)uiToPair(val, mapHeight).second;
}

bool deathZone(State st) {
    ui row[mapWidth], col[mapHeight];
    for (int i=0; i<mapWidth; i++) row[i] = 0;
    for (int i=0; i<mapHeight; i++) col[i] = 0;

    for (auto box : st.boxes) {
        ui x, y;
        std::tie(x, y) = uiToPair(box, mapHeight);
        row[x]++; col[y]++; 
    }

    // top, down
    for (int i=0; i<=1; i++) {
        if (row[boundary[i].first] < boundary[i].second) return true;
    }

    // left, right
    for (int i=2; i<=3; i++) {
        if (col[boundary[i].first] < boundary[i].second) return true;
    }

    return false;
}

void constructBoundary() {
    // top
    for (int i=0; i<mapWidth; i++) {
        bool avail = false;
        int cnt = 0;
        for (int j=0; j<mapHeight; j++) {
            ui pos = pairToUi(i, j, mapHeight);
            if (gameMap[pos] != wall) avail = true;
            if (gameMap[pos] == targetTile) cnt++;
        }
        if (avail) {
            boundary[top] = make_pair(i, cnt); 
            break;
        }
    }

    // down 
    for (int i=mapWidth-1; i>=0; i--) {
        bool avail = false;
        int cnt = 0;
        for (int j=0; j<mapHeight; j++) {
            ui pos = pairToUi(i, j, mapHeight);
            if (gameMap[pos] != wall) avail = true;
            if (gameMap[pos] == targetTile) cnt++;
        }
        if (avail) {
            boundary[down] = make_pair(i, cnt); 
            break;
        }
    }

    // left
    for (int j=0; j<mapHeight; j++) {
        bool avail = false;
        int cnt = 0;
        for (int i=0; i<mapWidth; i++) {
            ui pos = pairToUi(i, j, mapHeight);
            if (gameMap[pos] != wall) avail = true;
            if (gameMap[pos] == targetTile) cnt++;
        }
        if (avail) {
            boundary[left] = make_pair(j, cnt); 
            break;
        }
    }

    // right
    for (int j=mapHeight-1; j>=0; j--) {
        bool avail = false;
        int cnt = 0;
        for (int i=0; i<mapWidth; i++) {
            ui pos = pairToUi(i, j, mapHeight);
            if (gameMap[pos] != wall) avail = true;
            if (gameMap[pos] == targetTile) cnt++;
        }
        if (avail) {
            boundary[right] = make_pair(j, cnt); 
            break;
        }
    }
}

bool outOfBound(ui x, ui y, int move[]) {
    x += move[0]; y += move[1];
    if (x < 0 || x >= mapWidth || y < 0 || y >= mapHeight) return true;
    return false;
}


// check the state is finished 
bool done(State st) {
    for (auto box : st.boxes) {
        if (gameMap[box] != targetTile) return false; 
    }
    return true;
}

vector<State> tryMove(State st) {
    vector<State> nsts;
    // {box's new pos, box's original pos}
    unordered_map<ui, ui, boost::hash<ui> > candidate;
    unordered_map<ui, ui, boost::hash<ui> > repeatedCandidate;

    ui dir = 4;
    int prMove[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    int uiMove[4] = {1, -1, mapHeight, -mapHeight};

    // list the potential candidate pos
    for (auto box : st.boxes) {
        for (int i=0; i<dir; i++) {
            ui bx, by;
            tie(bx, by) = uiToPair(box, mapHeight);
            if (outOfBound(bx, by, prMove[i])) continue;

            ui nb = box + uiMove[i];
            if (gameMap[nb] == wall || gameMap[nb] == fragileTile || st.boxes.find(nb) != st.boxes.end()) continue;

            ui px, py;
            tie(px, py) = uiToPair(nb, mapHeight);
            if (outOfBound(px, py, prMove[i])) continue;

            ui np = nb + uiMove[i];
            if (gameMap[np] == wall || st.boxes.find(np) != st.boxes.end()) continue;

            if (candidate.find(nb) != candidate.end()) {
                repeatedCandidate[nb] = box;
            }
            else candidate[nb] = box;
        }
    }

    // from the previous position, find out all the reachable pos in the candidate
    unordered_set<ui, boost::hash<ui>> reachable;
    unordered_set<ui, boost::hash<ui>> repeatedReachable;
    if (st.seq.size() == 0) {
        for (auto c : candidate) {
            reachable.insert(c.first);
        }
        for (auto c : repeatedCandidate) {
            repeatedReachable.insert(c.first);
        }
    }
    else {
        ui from = st.seq.back().first, to = st.seq.back().second;
        ui pre = to + (to - from);
        int cnt = 0;
        if (candidate.find(pre) != candidate.end()) {
            reachable.insert(pre);
            if (repeatedCandidate.find(pre) != repeatedCandidate.end()) 
                repeatedReachable.insert(pre);
            cnt++;
        }

        unordered_set<ui, boost::hash<ui>> vis;
        queue<ui> qu;

        qu.push(pre);
        vis.insert(pre);
        
        while (!qu.empty() && cnt < candidate.size()) {
            ui pos = qu.front();
            qu.pop();

            for (int i=0; i<dir; i++) {
                ui npos = pos + uiMove[i];
                if (vis.find(npos) != vis.end()) continue;
                vis.insert(npos);

                if (gameMap[npos] == wall || st.boxes.find(npos) != st.boxes.end()) continue;

                if (candidate.find(npos) != candidate.end()) {
                    reachable.insert(npos);
                    if (repeatedCandidate.find(npos) != repeatedCandidate.end()) 
                        repeatedReachable.insert(npos);
                    cnt++;
                }
                qu.push(npos);
            }
        }
    }

    for (auto nx : reachable) {
        unordered_set<ui, boost::hash<ui>> ust = st.boxes;
        ust.erase(candidate[nx]);
        ust.insert(nx);

        vector<Pair> vec = st.seq;
        vec.push_back(make_pair(candidate[nx], nx));
        State nst(ust, vec);
        nsts.push_back(nst);
    } 

    for (auto nx : repeatedReachable) {
        unordered_set<ui, boost::hash<ui>> ust = st.boxes;

        ust.erase(repeatedCandidate[nx]);
        ust.insert(nx);

        vector<Pair> vec = st.seq;
        vec.push_back(make_pair(repeatedCandidate[nx], nx));
        State nst(ust, vec);
        nsts.push_back(nst);
    }

    return nsts;
}

bool constructSolution(State st) {
    int prMove[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    unordered_map<int, char> moves = {
        {1, 'D'},
        {-1, 'A'},
        {mapHeight, 'S'},
        {-mapHeight, 'W'},
    };
    string solution;
    ui src, dst;
    src = endPlayerPos;
    unordered_set<ui, boost::hash<ui>> boxLoc = targetPos;

    bool visitable = false;
    for (int i=st.seq.size()-1; i>=0; i--) {
        char ch;
        ui from, to;
        tie(from, to) = st.seq[i];
        dst = to + (to - from);
        ch = moves[from - to];
        
        pair<ui, string> start = make_pair(src, "");
        queue< pair<ui, string> > qu;
        unordered_set<ui, boost::hash<ui>> visited;
        qu.push(start);
        visited.insert(start.first);

        while (!qu.empty()) {
            pair<ui, string> q = qu.front();
            qu.pop();

            if (q.first == dst) {
                boxLoc.erase(to);
                boxLoc.insert(from);
                solution += q.second;
                solution += ch;
                src = to;
                visitable = true;
                break;
            }

            ui idx = 0;
            for (auto mv : moves) {
                ui x, y;
                tie(x, y) = uiToPair(q.first, mapHeight);
                if (outOfBound(x, y, prMove[idx])) continue;
                idx++;
                
                ui nxt = q.first + mv.first;
                if (visited.find(nxt) != visited.end()) continue;
                if (gameMap[nxt] == wall || boxLoc.find(nxt) != boxLoc.end()) continue;

                qu.push(make_pair(nxt, q.second + mv.second));
                visited.insert(nxt);
            }
        }

        if (i == st.seq.size()-1 && !visitable) return false;
    }

    cout << solution << endl;
    return true;
}



int main(int argc, char *argv[]) {
    if (argc < 2) {
        cout << "not enough arguments!\n";
        return 0;
    }

    // read the input file & construct the gameMap
    unordered_set<ui, boost::hash<ui>> initBoxPos;
    readMap(argv[1], initBoxPos);
    constructBoundary();

    // data structure
    unordered_set<State, StateHash> visited;
    queue<State> qu;

    // start state
    qu.push(State(initBoxPos, vector<Pair>()));

    // BFS
    bool cancel = false;

    while (!qu.empty() && !cancel) {
        #pragma omp parallel for
        for (int i=0; i<qu.size(); i++) {
            if (!cancel) {
                State st;

                #pragma omp critical 
                {
                    st = qu.front();
                    qu.pop();
                }

                if (!cancel && done(st)) {
                    #pragma omp critical 
                    {
                        if (constructSolution(st)) {
                            cancel = true;
                        }
                    }
                }

                else {
                    vector<State> nsts = tryMove(st);
                    for (auto nst : nsts) {
                        #pragma omp critical 
                        {
                            if (visited.find(nst) == visited.end()) {
                                    visited.insert(nst);
                                    if (!deathZone(nst)) qu.push(nst);
                            }
                        }
                    }

                }

            }
        }
    }



    return 0;
}