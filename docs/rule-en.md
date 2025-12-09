## Korean Janggi Rules (Formalized for AI Implementation)

This document describes the rules of Korean Janggi in a way that **an AI can read and implement directly**.  
The language is natural, but written to be **as formally interpretable as possible**.

This document operates on two layers at once:

- **Standard rules**: Rules actually used in real Korean Janggi (mainly Sections 1–7).
- **Project rules/options**: Fixed initial setup, scoring (+1.5), and simplifications/constraints used **only in this project’s AI engine** (Sections 3.3, 8, 9, etc.).

Whenever the text explicitly says “project rules/options”, remember that those are **not official Janggi rules,  
but policies chosen for this engine**.

---

## 1. Board and Coordinate System

- **Board size**

  - 9 files × 10 ranks = **90 intersections**.
  - In Janggi, pieces are placed **on line intersections**, not inside squares (unlike chess).

- **Palaces (궁성, Palace)**

  - Each side (Han at the bottom, Cho at the top) has a 3×3 palace in the center of its home area.
  - Inside each palace there are diagonal lines, and these diagonal lines are also **valid movement paths** for certain pieces.

- **Coordinate system definition (recommended example)**
  - Files (columns) from left to right: `a b c d e f g h i` (9 files).
  - Ranks (rows) from bottom to top: `1 2 ... 10` (10 ranks).
  - **Han (bottom side)**: “Forward” = increasing rank (e.g., `3 → 4`).
  - **Cho (top side)**: “Forward” = decreasing rank (e.g., `8 → 7`).

Example:

- `a1`: bottom-left corner from Han’s perspective.
- `e2`: center point of Han’s palace.
- `e9`: center point of Cho’s palace.

---

## 2. Piece Types

Each side (Han/Cho) has the following **16 pieces**:

- **궁 (King, shared name for Han/Cho)**: 1 piece (king role)
- **사 (Guard)**: 2 pieces
- **상 (Elephant)**: 2 pieces
- **마 (Horse)**: 2 pieces
- **차 (Rook)**: 2 pieces
- **포 (Cannon)**: 2 pieces
- **졸/병 (Pawn)**: 5 pieces

For AI implementation, you can usually model them with:

- Side: `HAN`, `CHO`
- Piece type: `KING`, `GUARD`, `ELEPHANT`, `HORSE`, `ROOK`, `CANNON`, `PAWN`

---

## 3. Initial Setup (Exact Coordinates)

### 3.1 Han (bottom side, second to move)

Han pieces are placed as follows, assuming **view from Han’s side (looking upward)**:

- **Rank 1 (`a1` ~ `i1`)**

  - `a1`: Rook (`HAN_ROOK`)
  - `b1`: Horse (`HAN_HORSE`)
  - `c1`: Elephant (`HAN_ELEPHANT`)
  - `d1`: Guard (`HAN_GUARD`)
  - `e1`: (empty at the start)
  - `f1`: Guard (`HAN_GUARD`)
  - `g1`: Elephant (`HAN_ELEPHANT`)
  - `h1`: Horse (`HAN_HORSE`)
  - `i1`: Rook (`HAN_ROOK`)

- **Rank 2**

  - `e2`: King (`HAN_KING`) — center of Han’s palace

- **Rank 3**

  - `b3`: Cannon (`HAN_CANNON`)
  - `h3`: Cannon (`HAN_CANNON`)

- **Rank 4**

  - `a4`: Pawn (`HAN_PAWN`)
  - `c4`: Pawn (`HAN_PAWN`)
  - `e4`: Pawn (`HAN_PAWN`)
  - `g4`: Pawn (`HAN_PAWN`)
  - `i4`: Pawn (`HAN_PAWN`)

- **Han’s palace area**
  - Files: `d e f`
  - Ranks: `1 2 3`
  - That is, the 3×3 area `d1 ~ f3` is Han’s palace.

### 3.2 Cho (top side, first to move)

Cho’s setup is the **vertical mirror** of Han’s:

- **Rank 10 (`a10` ~ `i10`)**

  - `a10`: Rook (`CHO_ROOK`)
  - `b10`: Horse (`CHO_HORSE`)
  - `c10`: Elephant (`CHO_ELEPHANT`)
  - `d10`: Guard (`CHO_GUARD`)
  - `e10`: (empty at the start)
  - `f10`: Guard (`CHO_GUARD`)
  - `g10`: Elephant (`CHO_ELEPHANT`)
  - `h10`: Horse (`CHO_HORSE`)
  - `i10`: Rook (`CHO_ROOK`)

- **Rank 9**

  - `e9`: King (`CHO_KING`) — center of Cho’s palace

- **Rank 8**

  - `b8`: Cannon (`CHO_CANNON`)
  - `h8`: Cannon (`CHO_CANNON`)

- **Rank 7**

  - `a7`: Pawn (`CHO_PAWN`)
  - `c7`: Pawn (`CHO_PAWN`)
  - `e7`: Pawn (`CHO_PAWN`)
  - `g7`: Pawn (`CHO_PAWN`)
  - `i7`: Pawn (`CHO_PAWN`)

- **Cho’s palace area**
  - Files: `d e f`
  - Ranks: `8 9 10`
  - That is, the 3×3 area `d8 ~ f10` is Cho’s palace.

### 3.3 Initial Formation (Starting Setup) Explanation

- **Meaning of “상차림” (starting formation)**

  - “상차림” refers to **how pieces are arranged at the start of the game**.
  - In traditional rules, on the 1st rank, the outermost squares (`a`, `i`) always hold rooks,  
    and the three central palace-adjacent squares (`d`, `e`, `f`) hold Guard – (empty) – Guard.
  - The four in-between squares (`b`, `c`, `g`, `h`) are **for Horses and Elephants**.  
    The rule is that **only the order of Horses and Elephants on these four squares is flexible.**
  - The king does not start on rank 1, but at the center of its palace (`e2` for Han, `e9` for Cho).  
    Cannons are placed on the rank immediately in front of the king,  
    and pawns are on the rank in front of the cannons.

- **Freedom in Horse/Elephant arrangement (4 major formations)**

  - Depending on how the four squares (`b1, c1, g1, h1` / `b10, c10, g10, h10`) are filled,
    four representative formations are distinguished:
    - **Original formation** (also called `상마상마` or “Gui-ma” in some naming systems)
    - **Right Elephant formation** (also called `마상마상`, “Gui-ma” variant)
    - **Inner Elephant formation** (also called `마상상마`, “Wonyang-ma”)
    - **Outer Elephant formation** (also called `상마마상`, “Yang-gui-ma”)
  - The four-character notation in parentheses  
    **represents, from left to right, the order of Horse/Elephant on `b1, c1, g1, h1`.**
    - Example: `마상상마` (“Horse Elephant Elephant Horse”) means:
      - For Han: `b1=Horse, c1=Elephant, g1=Elephant, h1=Horse`.
  - Formations like **Horse Horse Elephant Elephant** or **Elephant Elephant Horse Horse**,  
    where both Horses or both Elephants are stuck on one side,  
    are considered unbalanced and are **not allowed by rule**.

- **Starting formation used in this document (project rule)**

  - This ruleset refers to the traditional formation concepts above, but  
    for implementation simplicity we use a **single fixed coordinate-based starting formation**:
    - On the 1st rank, the Horse/Elephant pattern is fixed to **Inner Elephant (`마상상마`)**.
      - Han: `b1=Horse, c1=Elephant, g1=Elephant, h1=Horse`
      - Cho: `b10=Horse, c10=Elephant, g10=Elephant, h10=Horse`
    - All other pieces are placed as follows:
      - Han: King at `e2` (rank 2), Cannons at `b3` and `h3` (rank 3), Pawns at  
        `a4, c4, e4, g4, i4` (rank 4).
      - Cho: King at `e9` (rank 9), Cannons at `b8` and `h8` (rank 8), Pawns at  
        `a7, c7, e7, g7, i7` (rank 7).
  - In other words, **the king starts at the exact center of the palace**,  
    **cannons are on the two squares directly in front of the king’s file**, and  
    **pawns are on the five squares of the next rank in a straight line.**

- **Handling of variant formations**
  - In real Janggi, there exist **local rules / variant formations**  
    that change cannon locations, number of pawn ranks, etc.
  - However, this document and AI engine support **only the fixed formation described above**,  
    and do not consider any other formation unless additional options are implemented.

---

## 4. Basic Rules (Turn, Movement, Capture)

- **Turn order**

  - Cho moves first.
  - Thereafter turns alternate by single moves: Cho → Han → Cho → Han → …

- **Movement and capture**

  - On each turn, you may move **exactly one piece**.
  - A piece moves according to its movement rules to a **vacant intersection**, or  
    it may **capture an enemy piece located on its destination intersection**.
  - If the destination intersection contains a **friendly piece**, the move is illegal.

- **Blocking**

  - Rooks, Cannons, Elephants, and Horses are subject to blocking by other pieces  
    along their movement paths.  
    For details, see the respective sections for each piece.

- **Protecting your own king**
  - After any move, if the resulting position leaves **your own king in check**  
    (i.e., able to be captured on the opponent’s next move), that move is **illegal**.
  - In other words, you may not “expose your own king to check”.

---

## 5. Piece Movement Rules

In the descriptions below, “one step” refers to movement to a single **adjacent intersection**.

### 5.1 King (궁, KING)

- **Movement area**

  - The King must remain **within its own 3×3 palace** at all times.
  - Any move that leaves the palace is illegal.

- **Movement pattern**

  - One step up, down, left, or right (4 orthogonal directions).
  - One step diagonally **along the palace’s diagonal lines** (inside the palace only).

- **Capture**

  - The king may capture any **enemy piece** on a square it can legally move to.

- **Face-to-face kings forbidden**
  - If both kings are on the **same file (same vertical line)** and
  - **no pieces exist between them**, that position is **illegal**.
  - Any move that creates or maintains such a state is illegal.

### 5.2 Guard (사, GUARD)

- **Movement area**

  - Guards must also remain **inside their own palace**.

- **Movement pattern**

  - Within the palace, Guards can move **one step forward, backward, left, or right** (orthogonal moves).
  - **Diagonal moves** are only allowed **when the Guard is positioned on an X-diagonal intersection point**.
    - The palace has X-shaped diagonal lines. Guards can only move diagonally **when standing on a point along these diagonals**.
    - Diagonal points: **4 corners + center (5 points total)**
    - Han's palace diagonal points: `d1`, `f1`, `e2` (center), `d3`, `f3`
    - Cho's palace diagonal points: `d8`, `f8`, `e9` (center), `d10`, `f10`
    - Example: A Guard at `e2` (center) can move diagonally to `d1`, `f1`, `d3`, or `f3`.
    - Example: A Guard at `d1` (corner) can only move diagonally to `e2` (center).
    - **Non-diagonal points in the palace** (e.g., `d2`, `e1`, `e3`, `f2`) **do not allow diagonal moves**.

- **Capture**

  - A Guard may capture an enemy piece on any square it can legally move to.

### 5.3 Elephant (상, ELEPHANT)

- **Movement area**

  - Elephants can move **anywhere on the board**, inside or outside palaces.

- **Movement pattern (1 orthogonal step + 2 diagonal steps in same direction)**

  - First move **one step orthogonally** (up, down, left, or right),
  - Then from there move **two steps diagonally** in a direction that **extends from the orthogonal direction**.
  - **Important**: The diagonal direction must **share a component with the orthogonal direction**:
    - If orthogonal is right (→): diagonal must be **up-right (↗)** or **down-right (↘)**
    - If orthogonal is left (←): diagonal must be **up-left (↖)** or **down-left (↙)**
    - If orthogonal is up (↑): diagonal must be **up-right (↗)** or **up-left (↖)**
    - If orthogonal is down (↓): diagonal must be **down-right (↘)** or **down-left (↙)**
  - Overall, this is a **3-step extended L-shaped move**.
  - Possible final destinations (relative to starting position):
    - (+3, +2), (+3, -2), (-3, +2), (-3, -2)
    - (+2, +3), (+2, -3), (-2, +3), (-2, -3)
  - Along the path:
    - The first orthogonal step,
    - And the next diagonal step  
      — if **either** of these intermediate squares contains any piece (friendly or enemy),  
      the move is **blocked** and therefore illegal.

  **Note**: These rules reflect **standard Korean Janggi**.  
  (In some variants or in Chinese Xiangqi, the Elephant's movement may be defined  
  as simply moving two diagonals, but this document follows the Korean standard.)

- **Capture**

  - The Elephant may capture an enemy piece on its final destination square.

### 5.4 Horse (마, HORSE)

- **Movement area**

  - Horses can move **anywhere on the board**.

- **Movement pattern (1 orthogonal step + 1 diagonal step)**

  - First move **one step orthogonally** (up, down, left, or right),
  - Then from there move **one step diagonally**.
  - This is similar to the knight’s `L`-shaped movement in chess, but  
    if the **first orthogonal square is occupied by any piece**, the move is blocked  
    (this is often called the Horse’s “leg” being blocked).

- **Capture**

  - The Horse captures an enemy piece on its final destination square.

### 5.5 Rook (차, ROOK)

- **Movement area**

  - The Rook can move on the **entire board**, including the **palace diagonals**.

- **Movement pattern**

  - Any number of squares **orthogonally** (up, down, left, right).
  - There must be **no pieces** between the starting and ending square (no jumping).
  - Inside the palace, the diagonals are also treated as **straight lines**,  
    so the Rook can move along palace diagonals continuously.

- **Capture**

  - The Rook captures an enemy piece on its destination square.

### 5.6 Cannon (포, CANNON)

The Cannon is a characteristic Janggi piece that **must always jump over exactly one “screen” piece**.

- **Common rules**

  - It moves only in **straight lines** (orthogonal directions) and along **palace diagonals**.
  - Between its start and end squares there must be **exactly one piece** (the **screen**).
  - This screen piece must be **a non-Cannon piece**.  
    (Cannons **cannot** use another Cannon as a screen.)
  - If there are any additional pieces, beyond this single screen, between start and end,  
    the move is illegal.

- **Move to an empty square**

  - The destination must be **empty**.
  - Between start and destination there must be **exactly one non-Cannon piece**.  
    → legal movement.

- **Capture**

  - The destination square must contain an **enemy piece**.
  - Between start and destination there must be **exactly one non-Cannon piece**.  
    → legal capture.

### 5.7 Pawn (졸/병, PAWN)

- **Basic forward direction**

  - Han: “Forward” = rank **+1** (number increasing, moving upward).
  - Cho: “Forward” = rank **-1** (number decreasing, moving downward).

- **Movement pattern**

  - Normally moves **one step forward**, or **one step sideways (left/right)**.
  - It **can never move backward** in any situation.
  - **Special rules inside the palace**:
    - When a Pawn is inside a palace:
      - In addition to its normal moves (one step forward, or one step left/right),
      - It may also move **one step diagonally forward** along the palace diagonals  
        (forward-left or forward-right).
    - However, it may **not move diagonally backward** along those diagonals.
  - **Upon reaching the farthest rank of the enemy side**:
    - When a Han Pawn reaches rank 10, or a Cho Pawn reaches rank 1:
      - It can **no longer move forward or diagonally forward**, and
      - It is restricted to **moving only left or right by one step**.

- **Capture**

  - The Pawn captures an enemy piece occupying any square it can legally move to.

---

## 6. Check, Checkmate, and Stalemate

### 6.1 Check

- After a player’s move, if the opponent’s king becomes **immediately capturable on the next turn**,  
  we say that player has “given check”.
- The checked player must, on **their very next turn**:
  - Move the king to a safe square, or
  - Capture the attacking piece, or
  - Block the attack path with another piece,  
  - So that in the resulting position **their own king is no longer under attack**.

### 6.2 Checkmate

Checkmate occurs when both of the following conditions hold:

1. It is **the current player’s turn**.
2. In the current position, the player has **no legal moves at all**, and at the same time  
   **their king is in check**.

In this case the **current player loses**, and the opponent wins.

### 6.3 Stalemate

In Janggi, stalemate also counts as a **loss**.

- It is **the current player’s turn**, and
- In the current position, the player has **no legal moves**, and
- At the same time, **their king is not in check**.

Then:

- The position is a stalemate, and
- **The side to move loses**, the opponent wins.

In summary, in Janggi:

- You win by capturing the opponent’s king, or
- By putting the opponent in a state where they **have no legal moves**  
  (whether they are in check or not).

---

## 7. Repetition, Perpetual Check, and Draws

### 7.1 Repetition of the Same Position

- If the **same board position + same side to move** repeats **multiple times**,  
  the game is generally considered a **draw**.
- In actual tournaments, the exact number of repetitions needed for a draw may vary  
  (e.g., 3-fold repetition, etc.).

### 7.2 Perpetual Check (“Big Check”, 빅장)

- If one side **continually gives check repeatedly**  
  in such a way that the opponent cannot play meaningfully, this is called “Big Check” (빅장).
- In modern regulations this is usually **forbidden**, and
  - If the checking side refuses to change their sequence, they may be **ruled as losing**, or
  - Under some rule sets, the game may be declared a **draw**.

### 7.3 Other Draws

The following situations may also result in a draw, depending on tournament/platform rules:

- Both players agree to end the game without a decisive result.
- There is no meaningful progress (captures, advances, attacks, etc.) for a very long sequence of moves.
- Special rules combining repetition, perpetual check, and similar situations.

---

## 8. Piece Values and Scoring System

Based on a typical reference table (e.g., from annotated diagrams),  
the **standard piece values** are as follows:

- **King (궁, KING)**: no numerical value (`-`)
  - Usually excluded from score calculations, or treated as having **infinite value** in theory.
- **Rook (차, ROOK)**: 13 points × 2 = 26 points
- **Cannon (포, CANNON)**: 7 points × 2 = 14 points
- **Horse (마, HORSE)**: 5 points × 2 = 10 points
- **Elephant (상, ELEPHANT)**: 3 points × 2 = 6 points
- **Guard (사, GUARD)**: 3 points × 2 = 6 points
- **Pawn (졸/병, PAWN)**: 2 points × 5 = 10 points

Summing these gives **72 points per side** (excluding the king).

- **Total piece value**: 72.0 points
- In many practical rule sets, the **second player (the one who moves second)**  
  receives an additional **+1.5 points**.
  - This fractional 1.5 is used as a **draw-avoidance mechanism** (similar to komi in Go).
  - In this document, **Cho is always first player**, **Han is always second player**,  
    so the **+1.5 adjustment is applied to Han**.
  - Example:
    - Cho: 72.0 points
    - Han: 73.5 points

For AI/engine scoring:

- If the game ends by **checkmate or stalemate**, the winner/loser is decided **by that result**,  
  regardless of material scores.
- If neither king is captured and only pieces remain, or
  if the game ends regardless of check (repetition, perpetual check, agreement, etc.), then  
  the remaining piece values + second-player adjustment (+1.5) can be used  
  to determine win/loss/draw.

**Note**: This scoring system and the +1.5 adjustment are **not a mandatory core part of official Korean Janggi rules**.  
They are more like an **evaluation/anti-draw option** used in this project and some platforms.

---

## 9. Summary from an AI Implementation Perspective

### 9.1 State Representation

- Required elements
  - 9×10 board of intersections, with per-square piece information (empty / (side, type)).
  - Side to move (`HAN` or `CHO`).
- Optional elements
  - Position hash for repetition detection (e.g., Zobrist hash).
  - Metadata such as repetition counts, perpetual check flags, etc.

### 9.2 Legal Move Generation

1. For every piece belonging to the side to move,  
   generate a **list of candidate moves** according to the movement rules defined above.
2. For each candidate move:
   - Apply the move on a copy of the current state, and
   - Check whether **your own king is in check** in the resulting state.
3. Any move that leaves your king in check in the resulting position  
   must be **removed from the list of legal moves**.

### 9.3 Game Termination Detection

1. Count the number of **legal moves** available to the side to move.
2. Determine whether the side to move is currently in check.
3. Process the result as follows:
   - Legal moves = 0 & in check = yes → **Checkmate (side to move loses)**.
   - Legal moves = 0 & in check = no → **Stalemate (side to move loses)**.
   - Otherwise → Game continues.


