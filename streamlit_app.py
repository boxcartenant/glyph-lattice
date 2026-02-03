import streamlit as st
import random
import numpy as np
import json
import hashlib
from collections import defaultdict
import matplotlib.pyplot as plt

# Define possible upgrades for each root and branch
possible_upgrades = {
    'shape': {
        'linear': ['steep', 'shallow', '45deg'],
        'trig': ['sine', 'cosine', 'tangent', 'sintan', 'costan', 'tantan', 'vert', 'horiz'],
        'poly': ['order2pos', 'order2neg', 'order3pos', 'order3neg'],
        'log': ['asympt'],
    },
    'distribution': {
        'bell': ['basic', 'narrow', 'wide'],
        'parab': ['basic', 'sharp'],
        'sine': ['basic', 'highfreq'],
    },
    'coordinate': {
        'cartesian': ['curvilinear', 'holomorphic'],
        'polar': ['basic', 'bipolar', 'elliptic'],
        'barycentric': ['3sided', 'nsided', 'meanvalue', 'bezier'],
        'parabolic': ['basic', 'confocal', 'hamilton'],
    },
    'prioritization': {
        'fixed': ['NE', 'NW', 'SE', 'SW', 'center'],
        'historical': ['prior_turn', 'two_prior', 'enemy_prior', 'enemy_two'],
        'enemy_weighted': ['most_enemies', 'least', 'improved'],
    },
    'collision': {
        'prefer': ['basic', 'enemies', 'allies'],
        'avoid': ['basic', 'enemies', 'allies'],
        'passover': ['n1', 'n2', 'n3'],
    },
}

class TechTree:
    def __init__(self):
        self.trees = {
            'shape': {
                'linear': ['cardinal'],  # default
                'trig': [],
                'poly': [],
                'log': [],
            },
            'distribution': {
                'random': ['random'],  # default
                'bell': [],
                'parab': [],
                'sine': [],
            },
            'coordinate': {
                'cartesian': ['basic'],
                'polar': [],
                'barycentric': [],
                'parabolic': [],
            },
            'prioritization': {
                'fixed': ['N', 'S', 'E', 'W'],
                'historical': [],
                'enemy_weighted': [],
            },
            'collision': {
                'basic': ['basic'],
                'prefer': [],
                'avoid': [],
                'passover': [],
            },
        }

    def get_unlocked_shapes(self):
        return [u for b in self.trees['shape'].values() for u in b]

    def get_unlocked_dists(self):
        return [u for b in self.trees['distribution'].values() for u in b]

    def get_unlocked_prios(self):
        return [u for b in self.trees['prioritization'].values() for u in b]

    def get_available_upgrades(self, root, branch):
        if root in possible_upgrades and branch in possible_upgrades[root]:
            return [u for u in possible_upgrades[root][branch] if u not in self.trees[root][branch]]
        return []

# Functions for game logic
def get_placements(owner, glyph_type, shape, dist, prio, board):
    positions = []
    if shape == 'cardinal' and dist == 'random' and prio in ['N', 'S', 'E', 'W']:
        if prio == 'N':
            y = 0
        elif prio == 'S':
            y = 18
        elif prio == 'E':
            x = 18
        elif prio == 'W':
            x = 0
        if prio in ['N', 'S']:
            path = [(i, y) for i in range(19)]
        else:
            path = [(x, i) for i in range(19)]
        indices = np.arange(19)
        chosen = np.random.choice(indices, size=8, replace=False)
        positions = [path[idx] for idx in chosen]
    else:
        # Placeholder for other shapes/dist/prio
        for _ in range(8):
            positions.append((random.randint(0, 18), random.randint(0, 18)))
    return positions

def get_shape_curve(shape, prio):
    if shape == 'cardinal':
        if prio == 'N':
            x_vals = np.linspace(0, 18, 100)
            y_vals = np.zeros(100)
        elif prio == 'S':
            x_vals = np.linspace(0, 18, 100)
            y_vals = np.full(100, 18)
        elif prio == 'E':
            x_vals = np.full(100, 18)
            y_vals = np.linspace(0, 18, 100)
        elif prio == 'W':
            x_vals = np.zeros(100)
            y_vals = np.linspace(0, 18, 100)
        else:
            x_vals = np.linspace(0, 18, 100)
            y_vals = np.linspace(0, 18, 100)  # placeholder
    else:
        # Placeholder curve, e.g., a wave
        x_vals = np.linspace(0, 18, 100)
        if prio == 'S':
            y_vals = 18 - 4 * np.abs(np.sin(x_vals / 3))  # arcs near bottom
        else:
            y_vals = 9 + 4 * np.sin(x_vals / 3)
    return x_vals, y_vals

def resolve_collision(attacker_type, attacker_owner, defender_type, defender_owner):
    if defender_type == 'wall':
        return 'wall', None
    if attacker_type == defender_type:
        if attacker_type == 'd':
            return 'wall', None
        if attacker_type == 'e':
            return None, None
        if attacker_type == 'f':
            return None, None  # special handled outside
        # For other same types, keep defender
        return defender_type, defender_owner

    # Handle cases where 'e' is involved (priority as it changes ownership)
    if attacker_type == 'e':
        return defender_type, attacker_owner
    if defender_type == 'e':
        return attacker_type, defender_owner

    # Handle cases where 'f' is involved
    if attacker_type == 'f':
        if defender_type in ['a', 'b', 'c', 'd']:
            return 'f', attacker_owner
    if defender_type == 'f':
        if attacker_type in ['a', 'b', 'c', 'd']:
            return 'f', defender_owner

    # Handle cases where 'd' is involved
    if attacker_type == 'd':
        if defender_type in ['a', 'b']:
            return 'd', defender_owner
    if defender_type == 'd':
        if attacker_type in ['a', 'b']:
            return 'd', attacker_owner

    # Handle cases where 'c' is involved
    if attacker_type == 'c':
        if defender_type == 'a':
            return 'b', defender_owner
        if defender_type in ['b', 'c', 'd']:
            return None, None
    if defender_type == 'c':
        if attacker_type == 'a':
            return 'b', attacker_owner
        if attacker_type in ['b', 'c', 'd']:
            return None, None

    # Default: keep defender (cannot place on occupied without specific rule)
    return defender_type, defender_owner

def capture_groups(board, owners):
    visited = np.zeros((19, 19), dtype=bool)
    for i in range(19):
        for j in range(19):
            if board[i, j] != '.' and not visited[i, j]:
                group, liberties = get_group_liberties(i, j, board, owners, visited)
                if liberties == 0:
                    for gx, gy in group:
                        board[gx, gy] = '.'
                        owners[gx, gy] = None

def get_group_liberties(start_x, start_y, board, owners, visited):
    owner = owners[start_x, start_y]
    group = []
    liberties = set()
    stack = [(start_x, start_y)]
    visited[start_x, start_y] = True
    while stack:
        x, y = stack.pop()
        group.append((x, y))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 19 and 0 <= ny < 19:
                if board[nx, ny] == '.':
                    liberties.add((nx, ny))
                elif owners[nx, ny] == owner and board[nx, ny] == board[start_x, start_y] and not visited[nx, ny]:  # Same type for now
                    visited[nx, ny] = True
                    stack.append((nx, ny))
    return group, len(liberties)

def end_stage(game_state):
    board = game_state['board']
    owners = game_state['owners']
    visited = np.zeros((19, 19), dtype=bool)
    player_territory = 0
    pc_territory = 0
    for i in range(19):
        for j in range(19):
            if board[i, j] == '.' and not visited[i, j]:
                region, border_owners = get_region(i, j, board, owners, visited)
                if len(region) > 49:
                    continue
                unique_borders = set(border_owners)
                if len(unique_borders) == 1 and is_enclosed(region):
                    own = list(unique_borders)[0]
                    size = len(region)
                    if own == 'player':
                        player_territory += size
                    elif own == 'pc':
                        pc_territory += size
    # Add type a cells
    for i in range(19):
        for j in range(19):
            if board[i, j] == 'a':
                if owners[i, j] == 'player':
                    player_territory += 1
                elif owners[i, j] == 'pc':
                    pc_territory += 1
    st.write(f"Player territory: {player_territory}")
    st.write(f"PC territory: {pc_territory}")
    coin = game_state['coin']
    coin += 10 * player_territory
    if player_territory > pc_territory:
        game_state['stage'] += 1
        game_state['losses'] = 0
    else:
        game_state['losses'] += 1
    game_state['coin'] = coin
    game_state['stage_started'] = False
    game_state['board'] = np.full((19, 19), '.', dtype='<U1')
    game_state['owners'] = np.full((19, 19), None, dtype=object)
    game_state['hand'] = None
    game_state['current_turn'] = 0
    game_state['pre_collision_player'] = None
    game_state['pre_collision_pc'] = None
    game_state['selected_glyph'] = None
    game_state['selected_shape'] = None
    game_state['selected_dist'] = None
    game_state['selected_prio'] = None

def get_region(start_x, start_y, board, owners, visited):
    region = []
    border_owners = []
    stack = [(start_x, start_y)]
    visited[start_x, start_y] = True
    while stack:
        x, y = stack.pop()
        region.append((x, y))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 19 and 0 <= ny < 19:
                if board[nx, ny] == '.' and not visited[nx, ny]:
                    visited[nx, ny] = True
                    stack.append((nx, ny))
                elif board[nx, ny] != '.':
                    border_owners.append(owners[nx, ny])
    return region, border_owners

def is_enclosed(region):
    for x, y in region:
        if x == 0 or x == 18 or y == 0 or y == 18:
            return False
    return True

def start_stage(game_state):
    game_state['stage_started'] = True
    stage = game_state['stage']
    game_seed = game_state['game_seed']
    stage_seed_str = str(game_seed) + str(stage)
    stage_seed = int(hashlib.sha256(stage_seed_str.encode()).hexdigest(), 16) % 10**10
    game_state['stage_seed'] = stage_seed
    random.seed(stage_seed)
    pc_glyphs = {g: TechTree() for g in 'abcdef'}
    num_upgrades = 5 + (stage // 5)
    for _ in range(num_upgrades):
        glyph = random.choice(list('abcdef'))
        root = random.choice(list(pc_glyphs[glyph].trees.keys()))
        branch = random.choice(list(pc_glyphs[glyph].trees[root].keys()))
        available = pc_glyphs[glyph].get_available_upgrades(root, branch)
        if available:
            upgrade = random.choice(available)
            pc_glyphs[glyph].trees[root][branch].append(upgrade)
    game_state['pc_glyphs'] = pc_glyphs  # Save pc_glyphs in state

# Main app
if 'game_state' not in st.session_state:
    st.session_state.game_state = {
        'stage': 1,
        'losses': 0,
        'coin': 0,
        'board': np.full((19, 19), '.', dtype='<U1'),
        'owners': np.full((19, 19), None, dtype=object),
        'glyphs': {g: TechTree() for g in 'abcdef'},
        'pc_glyphs': {g: TechTree() for g in 'abcdef'},
        'game_seed': random.randint(0, 10**6),
        'stage_seed': None,
        'current_turn': 0,
        'stage_started': False,
        'hand': None,
        'selected_glyph': None,
        'selected_shape': None,
        'selected_dist': None,
        'selected_prio': None,
        'pre_collision_player': None,
        'pre_collision_pc': None,
        'upgrade_options': None,
    }

game_state = st.session_state.game_state

# Sidebar
with st.sidebar:
    page = st.selectbox('Page', ['Main', 'Tech Tree'])
    if st.button('Save'):
        json_str = json.dumps(game_state, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))
        st.download_button('Download Save', json_str, 'game_save.json')
    upload = st.file_uploader('Load Save')
    if upload:
        loaded_state = json.load(upload)
        st.session_state.game_state = loaded_state
        st.rerun()

if page == 'Main':
    # 1. Flatten the NumPy array and wrap each element in a <div>
    # This creates 361 small div containers
    cells_html = "".join([f"<div>{char}</div>" for char in st.session_state.game_state['board'].flatten()])
    # 2. Define the styling
    # We use aspect-ratio: 1 / 1 to ensure the board stays square
    grid_style = """
    <style>
        .game-board {
            display: grid;
            grid-template-columns: repeat(19, 1fr);
            width: 80%;
            max-width: 300px; /* Adjust based on your preference */
            aspect-ratio: 1 / 1;
            border: 1px solid #333;
            margin: 10px 0;
        }
        .game-board div {
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Courier New', monospace;
            border: 0.5px solid rgba(0,0,0,0.1); /* Subtle grid lines */
            font-size: 1.2rem;
        }
    </style>
    """
    # 3. Render
    st.markdown(grid_style, unsafe_allow_html=True)
    st.markdown(f'<div class="game-board">{cells_html}</div>', unsafe_allow_html=True)
    
    if not game_state['stage_started']:
        if st.button('Start Next Stage'):
            start_stage(game_state)
            st.rerun()
        if game_state['losses'] == 5:
            if st.button('Change Seed'):
                game_state['stage_seed'] = random.randint(0, 10**10)
                game_state['losses'] = 0
                st.rerun()
    else:
        glyphs = game_state['glyphs']
        pc_glyphs = game_state['pc_glyphs']
        shape_cards = list(set(u for g in glyphs for u in glyphs[g].get_unlocked_shapes()))
        dist_cards = list(set(u for g in glyphs for u in glyphs[g].get_unlocked_dists()))
        glyph_cards = list('abcdef')
        deck = glyph_cards + shape_cards + dist_cards
        if game_state['hand'] is None:
            hand = [
                random.choice(glyph_cards),
                random.choice(shape_cards) if shape_cards else 'cardinal',
                random.choice(dist_cards) if dist_cards else 'random',
                random.choice(deck),
                random.choice(deck),
            ]
            game_state['hand'] = hand
        hand = game_state['hand']
        hand_glyphs = [c for c in hand if c in glyph_cards]
        hand_shapes = [c for c in hand if c in shape_cards]
        hand_dists = [c for c in hand if c in dist_cards]
        st.write('Hand:')
        st.write('Glyphs:', hand_glyphs)
        st.write('Shapes:', hand_shapes)
        st.write('Dists:', hand_dists)
        selected_glyph = st.selectbox('Select Glyph', options=hand_glyphs, key='sel_glyph')
        if selected_glyph:
            glyph_shapes = glyphs[selected_glyph].get_unlocked_shapes()
            available_shapes = [s for s in hand_shapes if s in glyph_shapes]
            selected_shape = st.selectbox('Select Shape', options=available_shapes, key='sel_shape')
            glyph_dists = glyphs[selected_glyph].get_unlocked_dists()
            available_dists = [d for d in hand_dists if d in glyph_dists]
            selected_dist = st.selectbox('Select Dist', options=available_dists, key='sel_dist')
            glyph_prios = glyphs[selected_glyph].get_unlocked_prios()
            selected_prio = st.selectbox('Select Prio', options=glyph_prios, key='sel_prio')
            game_state['selected_glyph'] = selected_glyph
            game_state['selected_shape'] = selected_shape
            game_state['selected_dist'] = selected_dist
            game_state['selected_prio'] = selected_prio
            if selected_shape and selected_dist and selected_prio:
                if st.button('Preview'):
                    positions = get_placements('player', selected_glyph, selected_shape, selected_dist, selected_prio, game_state['board'])
                    preview_board = game_state['board'].copy()
                    for px, py in positions:
                        preview_board[px, py] = selected_glyph.upper()
                    preview_str = '\n'.join(' '.join(row) for row in preview_board)
                    st.markdown(
                        f"""
                        <div style="overflow: auto; border: 1px solid #ccc; padding: 10px; width: 400px; height: 400px;">
                            <pre style="font-family: monospace; font-size: 10px; line-height: 10px; white-space: pre; margin: 0;">
Preview:
{preview_str}
                            </pre>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                if st.button('Preview Graph'):
                    # Get curve for shape
                    x_vals, y_vals = get_shape_curve(selected_shape, selected_prio)
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.plot(x_vals, [18 - y for y in y_vals], 'b-')
                    ax.set_xlim(0, 18)
                    ax.set_ylim(0, 18)
                    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                    ax.set_xticks(range(19))
                    ax.set_yticks(range(19))
                    ax.set_yticklabels([str(18 - tick) for tick in range(19)])
                    ax.set_aspect('equal')
                    st.pyplot(fig)
                if st.button('Commit'):
                    player_positions = get_placements('player', selected_glyph, selected_shape, selected_dist, selected_prio, game_state['board'])
                    game_state['pre_collision_player'] = player_positions
                    player_choice = selected_glyph + (selected_shape or '') + (selected_dist or '') + (selected_prio or '')
                    choice_hash = int(hashlib.sha256(player_choice.encode()).hexdigest(), 16)
                    stage_seed = game_state['stage_seed']
                    current_turn = game_state['current_turn']
                    turn_seed = int(hashlib.sha256((str(stage_seed) + str(current_turn) + str(choice_hash)).encode()).hexdigest(), 16) % 10**10
                    random.seed(turn_seed)
                    pc_glyph = random.choice(list('abcdef'))
                    pc_shapes = pc_glyphs[pc_glyph].get_unlocked_shapes()
                    pc_shape = random.choice(pc_shapes) if pc_shapes else 'cardinal'
                    pc_dists = pc_glyphs[pc_glyph].get_unlocked_dists()
                    pc_dist = random.choice(pc_dists) if pc_dists else 'random'
                    pc_prios = pc_glyphs[pc_glyph].get_unlocked_prios()
                    pc_prio = random.choice(pc_prios) if pc_prios else 'N'
                    pc_positions = get_placements('pc', pc_glyph, pc_shape, pc_dist, pc_prio, game_state['board'])
                    game_state['pre_collision_pc'] = pc_positions
                    # Resolve placements
                    board = game_state['board']
                    owners = game_state['owners']
                    new_placements = defaultdict(list)
                    for p in player_positions:
                        new_placements[tuple(p)].append(('player', selected_glyph))
                    for p in pc_positions:
                        new_placements[tuple(p)].append(('pc', pc_glyph))
                    for pos, incoming in new_placements.items():
                        x, y = pos
                        existing_type = board[x, y] if board[x, y] != '.' else None
                        existing_owner = owners[x, y]
                        # Resolve incoming first if multiple
                        current_type = None
                        current_owner = None
                        special_f = False
                        if len(incoming) == 1:
                            current_owner, current_type = incoming[0]
                        else:
                            # Resolve between incoming
                            attacker_owner, attacker_type = incoming[0]
                            for defender_owner, defender_type in incoming[1:]:
                                new_type, new_owner = resolve_collision(attacker_type, attacker_owner, defender_type, defender_owner)
                                if new_type is None:
                                    if attacker_type == 'f' and defender_type == 'f':
                                        special_f = True
                                    current_type = None
                                    break
                                attacker_type, attacker_owner = new_type, new_owner
                            current_type = attacker_type
                            current_owner = attacker_owner
                        if current_type is None:
                            if special_f:
                                # Handle 3x3 type c
                                for dx in range(-1, 2):
                                    for dy in range(-1, 2):
                                        nx, ny = x + dx, y + dy
                                        if 0 <= nx < 19 and 0 <= ny < 19 and board[nx, ny] == '.':
                                            board[nx, ny] = 'c'
                                            owners[nx, ny] = current_owner  # or random? Assume last attacker
                            continue
                        # Now resolve with existing
                        if existing_type is not None:
                            current_type, current_owner = resolve_collision(current_type, current_owner, existing_type, existing_owner)
                        if current_type is None:
                            if existing_type == 'f' and (existing_type == 'f' or current_type == 'f'):  # Check for f collision with existing
                                board[x, y] = '.'
                                owners[x, y] = None
                                for dx in range(-1, 2):
                                    for dy in range(-1, 2):
                                        nx, ny = x + dx, y + dy
                                        if 0 <= nx < 19 and 0 <= ny < 19 and board[nx, ny] == '.':
                                            board[nx, ny] = 'c'
                                            owners[nx, ny] = current_owner
                            else:
                                board[x, y] = '.'
                                owners[x, y] = None
                        else:
                            board[x, y] = current_type
                            owners[x, y] = current_owner
                    capture_groups(board, owners)
                    occupied = np.sum(board != '.') / 361
                    if occupied > 0.8:
                        end_stage(game_state)
                    else:
                        game_state['current_turn'] += 1
                        game_state['hand'] = None
                    st.rerun()
    # Buttons for pre/post views
    if game_state['pre_collision_player']:
        if st.button('Show Player Pre-Collision'):
            temp_board = game_state['board'].copy()
            for px, py in game_state['pre_collision_player']:
                temp_board[px, py] = game_state['selected_glyph'].upper()
            temp_str = '\n'.join(' '.join(row) for row in temp_board)
            st.markdown(
                f"""
                <div style="overflow: auto; border: 1px solid #ccc; padding: 10px; width: 400px; height: 400px;">
                    <pre style="font-family: monospace; font-size: 10px; line-height: 10px; white-space: pre; margin: 0;">
{temp_str}
                    </pre>
                </div>
                """,
                unsafe_allow_html=True
            )
        if st.button('Show PC Pre-Collision'):
            temp_board = game_state['board'].copy()
            for px, py in game_state['pre_collision_pc']:
                temp_board[px, py] = 'X'  # Placeholder for PC
            temp_str = '\n'.join(' '.join(row) for row in temp_board)
            st.markdown(
                f"""
                <div style="overflow: auto; border: 1px solid #ccc; padding: 10px; width: 400px; height: 400px;">
                    <pre style="font-family: monospace; font-size: 10px; line-height: 10px; white-space: pre; margin: 0;">
{temp_str}
                    </pre>
                </div>
                """,
                unsafe_allow_html=True
            )
        if st.button('Show Post-Collision'):
            # Main board is post
            pass

elif page == 'Tech Tree':
    st.write(f"Coin: {game_state['coin']}")
    glyphs = game_state['glyphs']
    for g in 'abcdef':
        st.write(f"Glyph {g}:")
        for root, branches in glyphs[g].trees.items():
            st.write(f"  {root}:")
            for branch, unlocks in branches.items():
                st.write(f"    {branch}: {unlocks}")
    upgrade_options = game_state['upgrade_options']
    cost = 10  # Assume cost 10
    if upgrade_options is None:
        if st.button('Buy Upgrade', disabled=game_state['coin'] < cost):
            game_state['coin'] -= cost
            options = []
            for _ in range(3):
                glyph = random.choice('abcdef')
                root = random.choice(list(glyphs[glyph].trees.keys()))
                branch = random.choice(list(glyphs[glyph].trees[root].keys()))
                available = glyphs[glyph].get_available_upgrades(root, branch)
                if not available:
                    continue
                upgrade = random.choice(available)
                # Downside
                all_unlocked = []
                for gg in 'abcdef':
                    for rr in glyphs[gg].trees:
                        for bb in glyphs[gg].trees[rr]:
                            for uu in glyphs[gg].trees[rr][bb]:
                                all_unlocked.append((gg, rr, bb, uu))
                downside = random.choice(all_unlocked) if all_unlocked else None
                options.append({
                    'glyph': glyph,
                    'root': root,
                    'branch': branch,
                    'upgrade': upgrade,
                    'downside': downside
                })
            game_state['upgrade_options'] = options
            st.rerun()
    else:
        st.write("Select one upgrade option:")
        for idx, opt in enumerate(upgrade_options):
            st.write(f"Option {idx+1}: Upgrade glyph {opt['glyph']} {opt['root']} {opt['branch']} with {opt['upgrade']}")
            if opt['downside']:
                d_g, d_r, d_b, d_u = opt['downside']
                st.write(f"Downside: 50% chance to remove {d_u} from glyph {d_g} {d_r} {d_b}")
            if st.button(f"Select Option {idx+1}"):
                glyphs[opt['glyph']].trees[opt['root']][opt['branch']].append(opt['upgrade'])
                if opt['downside'] and random.random() < 0.5:
                    d_g, d_r, d_b, d_u = opt['downside']
                    glyphs[d_g].trees[d_r][d_b].remove(d_u)
                game_state['upgrade_options'] = None
                st.rerun()
