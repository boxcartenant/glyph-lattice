import streamlit as st
import random
import numpy as np
import json
import hashlib
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import copy

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

# Implemented upgrades (only those with actual code logic)
implemented_upgrades = {
    'shape': {
        'linear': ['cardinal'],
        'trig': ['sine', 'cosine'],  # Assuming placeholders are implemented
    },
    'distribution': {
        'random': ['random'],
    },
    'coordinate': {
        'cartesian': ['basic'],
    },
    'prioritization': {
        'fixed': ['N', 'S', 'E', 'W'],
    },
    'collision': {
        'basic': ['basic'],
    },
}

class TechTree:
    def __init__(self, unlock_all=False):
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
        if unlock_all:
            for root in self.trees:
                for branch in self.trees[root]:
                    if root in implemented_upgrades and branch in implemented_upgrades[root]:
                        self.trees[root][branch].extend(implemented_upgrades[root][branch])

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

# Glyph descriptions
glyph_descriptions = {
    'a': 'Class A (Territory focus) | Type a: Territory under the glyph counts as captured (for all other stones, territory under the glyph does not count as captured).',
    'b': 'Class A (Territory focus) | Type b: Can connect to type b across gaps of one tile for enclosure.',
    'c': 'Class B (Removal focused) | Type c: On collision, converts type a to type b, mutually removes type b and class B.',
    'd': 'Class B (Removal focused) | Type d: On collision, converts class A to type d, converts type d to "wall" (space removed from board, cannot count as territory, treated as edge of board).',
    'e': 'Class C (Tactical focus) | Type e: On collision, changes ownership to match the user which dropped this type e. On collision with type e, mutually destroys glyph.',
    'f': 'Class C (Tactical focus) | Type f: On collision, converts class B and class A to type f. On collision with type f, creates 3x3 square of type c centered on collision. No chain-reactions permitted (glyphs placed due to explosions will only be placed on empty tiles).',
}

# Sample upgrade descriptions (expand as needed)
upgrade_descriptions = {
    'cardinal': 'Basic cardinal line (horizontal or vertical).',
    'steep': 'Steep diagonal lines.',
    'shallow': 'Shallow diagonal lines.',
    '45deg': '45-degree diagonal lines.',
    'sine': 'Sine wave shape.',
    'cosine': 'Cosine wave shape.',
    'tangent': 'Tangent curve shape.',
    # Add more as needed...
    'random': 'Uniform random distribution along the shape.',
    'basic': 'Basic version of the branch (e.g., for bell: weighted toward middle).',
    # Placeholders for others
}

def get_upgrade_desc(upgrade):
    return upgrade_descriptions.get(upgrade, f"Unlocks {upgrade} functionality in this branch.")

# Functions for game logic
def calculate_shape(shape, prio, seed=None):
    if seed is not None:
        random.seed(seed)
    rcenter = random.triangular(0,9,4.5)
    if shape == 'cardinal':
        t_min, t_max = 0, 19
        if prio == 'W':
            f = lambda t: (0 + t, 0.5+rcenter)
        elif prio == 'E':
            f = lambda t: (0 + t, 18.5-rcenter)
        elif prio == 'N':
            f = lambda t: (0.5+rcenter, 0 + t)
        elif prio == 'S':
            f = lambda t: (18.5-rcenter, 0 + t)
        else:
            f = lambda t: (0 + t, 0 + t)  # placeholder diagonal
    else:
        # Placeholder for trig, assume cosine
        t_min, t_max = 0, 18
        if prio == 'S':
            f = lambda t: (0.5 + t, 18.5 - 4 * abs(math.cos(t * math.pi / 9)))
        elif prio == 'N':
            f = lambda t: (0.5 + t, 0.5 + 4 * abs(math.cos(t * math.pi / 9)))
        else:
            f = lambda t: (0.5 + t, 9.5 + 4 * math.sin(t * math.pi / 9))
    return f, t_min, t_max

def get_shape_curve(shape, prio):
    f, t_min, t_max = calculate_shape(shape, prio)
    ts = np.linspace(t_min, t_max, 100)
    x_vals = [f(t)[0] for t in ts]
    y_vals = [f(t)[1] for t in ts]
    return x_vals, y_vals

def get_placements(owner, glyph_type, shape, dist, prio, board):
    f, t_min, t_max = calculate_shape(shape, prio)
    positions = []
    if dist == 'random':
        ts = [random.uniform(t_min, t_max) for _ in range(8)]
    else:
        # Placeholder for other dists, use uniform
        ts = [random.uniform(t_min, t_max) for _ in range(8)]
    seen = set()
    for t in ts:
        y_center, x_center = f(t)
        px = round(x_center - 0.5)
        py = round(y_center - 0.5)
        px = max(0, min(18, px))
        py = max(0, min(18, py))
        pos = (px, py)
        if pos not in seen:
            seen.add(pos)
            positions.append(pos)
    # If fewer than 8 due to dups, add more
    while len(positions) < 8:
        t = random.uniform(t_min, t_max)
        y_center, x_center = f(t)
        px = round(x_center - 0.5)
        py = round(y_center - 0.5)
        px = max(0, min(18, px))
        py = max(0, min(18, py))
        pos = (px, py)
        if pos not in seen:
            seen.add(pos)
            positions.append(pos)
    return positions

def resolve_collision(attacker_type, attacker_owner, defender_type, defender_owner, new_stones = False):
    if defender_type == 'wall':
        return 'w', None
    if attacker_type == defender_type:
        if attacker_type == 'd':
            return 'w', None
        if attacker_type == 'e':
            return None, None
        if attacker_type == 'f':
            return None, attacker_owner  # special handled outside
        if new_stones:
            return None, None
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

def capture_groups(board, owners, game_state):
    visited = np.zeros((19, 19), dtype=bool)
    for i in range(19):
        for j in range(19):
            if board[i, j] != '.' and not visited[i, j]:
                group, liberties = get_group_liberties(i, j, board, owners, visited)
                if liberties == 0:
                    group_owner = owners[group[0][0], group[0][1]]
                    if group_owner == 'player':
                        game_state['pc_captured'] += len(group)
                    elif group_owner == 'pc':
                        game_state['player_captured'] += len(group)
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
    territory_board = np.full((19, 19), '.', dtype=object):
                    continue
                unique_borders = set(border_owners)
                if len(unique_borders) == 1 and is_enclosed(region):
                    own = list(unique_borders)[0]
                    size = len(region)
                    # Subtract enemy stones in region
                    enemy_count = 0
                    for rx, ry in region:
                        if board[rx, ry] != '.' and owners[rx, ry] != own:
                            enemy_count += 1
                    size -= enemy_count
                    if own == 'player':
                        player_territory += size
                    elif own == 'pc':
                        pc_territory += size
                    # Mark territory on territory_board
                    mark_char = 'X'
                    for rx, ry in region:
                        territory_board[rx, ry] = mark_char
                        territory_owners[rx, ry] = own
    # Add type a cells
    for i in range(19):
        for j in range(19):
            if board[i, j] == 'a':
                if owners[i, j] == 'player':
                    player_territory += 1
                elif owners[i, j] == 'pc':
                    pc_territory += 1
    game_state['final_board'] = board.copy()
    game_state['final_owners'] = owners.copy()
    game_state['final_territory_board'] = territory_board
    game_state['final_territory_owners'] = territory_owners
    game_state['player_territory'] = player_territory
    game_state['pc_territory'] = pc_territory
    game_state['player_score'] = player_territory + game_state['pc_captured']
    game_state['pc_score'] = pc_territory + game_state['player_captured']
    coin = game_state['coin']
    if game_state['player_score'] > game_state['pc_score']:
        coin += game_state['player_score']
        game_state['stage'] += 1
        game_state['losses'] = 0
    else:
        game_state['losses'] += 1
    game_state['coin'] = coin
    game_state['stage_started'] = False
    game_state['board'] = np.full((19, 19), '.', dtype='
        document.cookie = "game_state={json_str}; path=/";
        
        """,
        height=0,
    )

# Function to load game state from cookie
def load_state_from_cookie():
    cookie = st.experimental_get_query_params().get('cookie', [None])[0]
    if cookie:
        loaded_state = json.loads(cookie)
        st.session_state.game_state = loaded_state

# Load from cookie on init
load_state_from_cookie()

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
        save_state_to_cookie()
        st.rerun()
    if st.button('Clear Cookie / Restart Game'):
        st.components.v1.html(
            """
            
            """,
            height=0,
        )
        st.session_state.game_state = {
            'stage': 1,
            'losses': 0,
            'coin': 0,
            'board': np.full((19, 19), '.', dtype='{char}'
    # Styling with colors
    grid_style = """
    
    """
    # Render
    st.markdown(grid_style, unsafe_allow_html=True)
    st.markdown(f'{cells_html}', unsafe_allow_html=True)
    
    if not game_state['stage_started']:
        if 'final_board' in game_state and game_state['final_board'] is not None:
            # Endgame readout
            st.write("End of Stage")
            # Final board state
            st.write("Final Board State:")
            final_cells_html = ""
            for py in range(19):
                for px in range(19):
                    char = game_state['final_board'][py, px]
                    owner = game_state['final_owners'][py, px]
                    class_name = ''
                    if owner == 'player':
                        class_name = 'player-cell'
                    elif owner == 'pc':
                        class_name = 'pc-cell'
                    final_cells_html += f'
{char}
'
            st.markdown(f'{final_cells_html}', unsafe_allow_html=True)

            # Territory map
            st.write("Territory Map (X for scored territory, . and w for unscored/wall):")
            territory_cells_html = ""
            for py in range(19):
                for px in range(19):
                    char = game_state['final_territory_board'][py, px]
                    owner = game_state['final_territory_owners'][py, px]
                    class_name = ''
                    if char == 'X' and owner == 'player':
                        class_name = 'player-territory'
                    elif char == 'X' and owner == 'pc':
                        class_name = 'pc-territory'
                    elif char == 'w':
                        class_name = 'wall-cell'
                    territory_cells_html += f'
{char}
'
            st.markdown(f'{territory_cells_html}', unsafe_allow_html=True)

            # Captured stones
            st.write(f"Player captured stones: {game_state['pc_captured']}")
            st.write(f"PC captured stones: {game_state['player_captured']}")

            # Total scores
            st.write(f"Player total score: {game_state['player_score']}")
            st.write(f"PC total score: {game_state['pc_score']}")

            # Coins earned
            if game_state['player_score'] > game_state['pc_score']:
                st.write(f"Coins earned: {game_state['player_score']}")
            else:
                st.write("No coins earned (loss)")

            # Reset button
            if st.button('Reset for Next Stage'):
                game_state['final_board'] = None
                game_state['final_owners'] = None
                game_state['final_territory_board'] = None
                game_state['final_territory_owners'] = None
                game_state['player_territory'] = 0
                game_state['pc_territory'] = 0
                game_state['player_captured'] = 0
                game_state['pc_captured'] = 0
                game_state['player_score'] = 0
                game_state['pc_score'] = 0
                start_stage(game_state)
                save_state_to_cookie()
                st.rerun()
        else:
            if st.button('Start Next Stage'):
                start_stage(game_state)
                save_state_to_cookie()
                st.rerun()
        if game_state['losses'] == 5:
            if st.button('Change Seed'):
                game_state['stage_seed'] = random.randint(0, 10**10)
                game_state['losses'] = 0
                save_state_to_cookie()
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
        selected_glyph = st.selectbox('Select Glyph', options=hand_glyphs, key='sel_glyph')
        if selected_glyph:
            glyph_shapes = glyphs[selected_glyph].get_unlocked_shapes()
            available_shapes = [s for s in hand_shapes if s in glyph_shapes]
            selected_shape = st.selectbox('Select Shape', options=available_shapes, key='sel_shape')
            glyph_dists = glyphs[selected_glyph].get_unlocked_dists()
            available_dists = [d for d in hand_dists if d in glyph_dists]
            selected_dist = st.selectbox('Select Dist', options=available_dists, key='sel_dist')
            glyph_prios = glyphs[selected_glyph].get_unlocked_prios()
            selected_prio = st.selectbox('Select Priority', options=glyph_prios, key='sel_prio')
            game_state['selected_glyph'] = selected_glyph
            game_state['selected_shape'] = selected_shape
            game_state['selected_dist'] = selected_dist
            game_state['selected_prio'] = selected_prio
            if selected_shape and selected_dist and selected_prio:
                if st.button('Preview'):
                    positions = get_placements('player', selected_glyph, selected_shape, selected_dist, selected_prio, game_state['board'])
                    preview_board = game_state['board'].copy()
                    for px, py in positions:
                        preview_board[py, px] = selected_glyph.upper()
                    cells_html = ""
                    for i in range(19):
                        for j in range(19):
                            char = preview_board[i, j]
                            class_name = ''
                            if game_state['owners'][i,j] == 'pc' and not char.isupper():
                                class_name = 'pc-cell'
                            elif char.isupper() or game_state['owners'][i,j] == 'player':
                                class_name = 'player-cell'
                            cells_html += f'
{char}
'
                    st.markdown(grid_style, unsafe_allow_html=True)
                    st.markdown(f'{cells_html}', unsafe_allow_html=True)
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
                    game_state['debug_log'] = []  # Clear debug log
                    player_positions = get_placements('player', selected_glyph, selected_shape, selected_dist, selected_prio, game_state['board'])
                    game_state['pre_collision_player'] = player_positions
                    game_state['pre_collision_player_glyph'] = selected_glyph
                    player_choice = selected_glyph + (selected_shape or '') + (selected_dist or '') + (selected_prio or '')
                    choice_hash = int(hashlib.sha256(player_choice.encode()).hexdigest(), 16)
                    stage_seed = game_state['stage_seed']
                    current_turn = game_state['current_turn']
                    turn_seed = int(hashlib.sha256((str(stage_seed) + str(current_turn) + str(choice_hash)).encode()).hexdigest(), 16) % 10**10
                    random.seed(turn_seed)
                    glyphs_list = list('abcdef')
                    t_glyph = random.triangular(0, 5, game_state['pc_glyph_mode'])
                    pc_glyph = glyphs_list[round(t_glyph)]
                    game_state['pre_collision_pc_glyph'] = pc_glyph
                    pc_shapes = sorted(pc_glyphs[pc_glyph].get_unlocked_shapes())
                    len_s = len(pc_shapes)
                    if len_s > 0:
                        t_shape = random.triangular(0, len_s - 1, game_state['pc_shape_mode_frac'] * (len_s - 1))
                        pc_shape = pc_shapes[round(t_shape)] if pc_shapes else 'cardinal'
                    else:
                        pc_shape = 'cardinal'
                    pc_dists = sorted(pc_glyphs[pc_glyph].get_unlocked_dists())
                    len_d = len(pc_dists)
                    if len_d > 0:
                        t_dist = random.triangular(0, len_d - 1, game_state['pc_dist_mode_frac'] * (len_d - 1))
                        pc_dist = pc_dists[round(t_dist)] if pc_dists else 'random'
                    else:
                        pc_dist = 'random'
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
                    game_state['last_collisions'] = set()
                    for pos, incoming in new_placements.items():
                        x, y = pos
                        existing_type = board[y, x] if board[y, x] != '.' else None
                        existing_owner = owners[y, x]
                        collided = False
                        # Resolve incoming first if multiple
                        current_type = None
                        current_owner = None
                        special_f = False
                        if len(incoming) > 1:
                            collided = True
                        if len(incoming) == 1:
                            current_owner, current_type = incoming[0]
                        else:
                            # Resolve between incoming
                            attacker_owner, attacker_type = incoming[0]
                            explosion_owner = attacker_owner
                            for defender_owner, defender_type in incoming[1:]:
                                new_type, new_owner = resolve_collision(attacker_type, attacker_owner, defender_type, defender_owner, True) #last bool indicates these are new stones
                                game_state['debug_log'].append(f"incoming collision: {attacker_owner} {attacker_type} {defender_owner} {defender_type}, result: {new_type} {new_owner}")
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
                                        if 0 <= nx < 19 and 0 <= ny < 19 and board[ny, nx] == '.':
                                            board[ny, nx] = 'c'
                                            owners[ny, nx] = attacker_owner  # Use attacker_owner for f + f
                                            game_state['last_collisions'].add((nx, ny))
                                game_state['last_collisions'].add((x, y))
                            continue
                        # Now resolve with existing
                        if existing_type is not None:
                            collided = True
                            new_type, new_owner = resolve_collision(current_type, current_owner, existing_type, existing_owner)
                            game_state['debug_log'].append(f"collision w existing: {current_type} {current_owner} {existing_type} {existing_owner}, result: {new_type} {new_owner}")
                            current_type = new_type  # Fixed typo
                            current_owner = new_owner
                        if current_type is None:
                            if existing_type == 'f' and (existing_type == 'f' or current_type == 'f'):  # Check for f collision with existing
                                collided = True
                                board[y, x] = '.'
                                owners[y, x] = None
                                for dx in range(-1, 2):
                                    for dy in range(-1, 2):
                                        nx, ny = x + dx, y + dy
                                        if 0 <= nx < 19 and 0 <= ny < 19 and board[ny, nx] == '.':
                                            board[ny, nx] = 'c'
                                            owners[ny, nx] = current_owner
                                            game_state['last_collisions'].add((nx, ny))
                                game_state['last_collisions'].add((x, y))
                            else:
                                board[y, x] = '.'
                                owners[y, x] = None
                        else:
                            board[y, x] = current_type
                            owners[y, x] = current_owner
                        if collided:
                            game_state['last_collisions'].add((x, y))
                    capture_groups(board, owners, game_state)
                    occupied = np.sum(board != '.') / 361
                    if occupied > 0.6:
                        end_stage(game_state)
                    else:
                        game_state['current_turn'] += 1
                        game_state['hand'] = None
                    save_state_to_cookie()
                    st.rerun()
    # Buttons for pre/post views
    if game_state['pre_collision_player']:
        if st.button('Show Player Pre-Collision'):
            temp_board = game_state['board'].copy()
            for px, py in game_state['pre_collision_player']:
                temp_board[py, px] = game_state['pre_collision_player_glyph'].upper()
            cells_html = ""
            for i in range(19):
                for j in range(19):
                    char = temp_board[i, j]
                    class_name = ''
                    if game_state['owners'][i,j] == 'pc' and not char.isupper():
                        class_name = 'pc-cell'
                    elif char.isupper() or game_state['owners'][i,j] == 'player':
                        class_name = 'player-cell'
                    cells_html += f'
{char}
'
            st.markdown(grid_style, unsafe_allow_html=True)
            st.markdown(f'{cells_html}', unsafe_allow_html=True)
        if st.button('Show PC Pre-Collision'):
            temp_board = game_state['board'].copy()
            for px, py in game_state['pre_collision_pc']:
                temp_board[py, px] = game_state['pre_collision_pc_glyph'].upper() if 'pre_collision_pc_glyph' in game_state else 'X'
            cells_html = ""
            for i in range(19):
                for j in range(19):
                    char = temp_board[i, j]
                    class_name = ''
                    if game_state['owners'][i,j] == 'player' and not char.isupper():
                        class_name = 'player-cell'
                    elif char.isupper() or game_state['owners'][i,j] == 'pc':
                        class_name = 'pc-cell'
                    cells_html += f'
{char}
'
            st.markdown(grid_style, unsafe_allow_html=True)
            st.markdown(f'{cells_html}', unsafe_allow_html=True)
        if st.button('Show Collisions'):
            temp_board = game_state['board'].copy()
            for px, py in game_state['last_collisions']:
                if temp_board[py, px] != '.':
                    temp_board[py, px] = temp_board[py, px].upper()
            cells_html = ""
            for py in range(19):
                for px in range(19):
                    char = temp_board[py, px]
                    owner = game_state['owners'][py, px]
                    class_name = ''
                    if owner == 'player':
                        class_name = 'player-cell'
                    elif owner == 'pc':
                        class_name = 'pc-cell'
                    cells_html += f'
{char}
'
            st.markdown(grid_style, unsafe_allow_html=True)
            st.markdown(f'{cells_html}', unsafe_allow_html=True)

    # Display debug log
    if game_state['debug_log']:
        st.write("Debug Log:")
        for log in game_state['debug_log']:
            st.write(log)

elif page == 'Tech Tree':
    st.write(f"Coin: {game_state['coin']}")
    glyphs = game_state['glyphs']
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
            save_state_to_cookie()
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
                save_state_to_cookie()
                st.rerun()

    # New interactive tech tree UI
    selected_glyph = st.selectbox('Select Glyph to View Tech Tree', options=list('abcdef'))
    if selected_glyph:
        st.write(f"**Glyph Description:** {glyph_descriptions.get(selected_glyph, 'No description available.')}")
        tech = glyphs[selected_glyph]
        for root in tech.trees:
            with st.expander(root.capitalize()):
                branch = st.selectbox(f'Select Branch for {root.capitalize()}', options=list(tech.trees[root].keys()), key=f'branch_{selected_glyph}_{root}')
                if branch:
                    unlocked = tech.trees[root][branch]
                    available = tech.get_available_upgrades(root, branch)
                    st.write('**Unlocked Upgrades:**')
                    for u in unlocked:
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(u, key=f'btn_{selected_glyph}_{root}_{branch}_{u}'):
                                st.session_state[f'detail_{selected_glyph}_{root}_{branch}_{u}'] = True
                        if st.session_state.get(f'detail_{selected_glyph}_{root}_{branch}_{u}', False):
                            st.write(get_upgrade_desc(u))
                            # Since upgrades are bought randomly, no direct upgrade button; perhaps note
                            st.write("This upgrade is already unlocked. Upgrades are purchased via random options above.")
                            if st.button('Hide Details', key=f'hide_{selected_glyph}_{root}_{branch}_{u}'):
                                st.session_state[f'detail_{selected_glyph}_{root}_{branch}_{u}'] = False
                    st.write('**Available Upgrades (Samples):**')
                    # Show a couple of samples from available
                    samples = available[:2] if available else ['Sample1', 'Sample2']  # Fallback to placeholders if none available
                    for u in samples:
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(u, key=f'btn_avail_{selected_glyph}_{root}_{branch}_{u}'):
                                st.session_state[f'detail_avail_{selected_glyph}_{root}_{branch}_{u}'] = True
                        if st.session_state.get(f'detail_avail_{selected_glyph}_{root}_{branch}_{u}', False):
                            st.write(get_upgrade_desc(u))
                            # Note: Cannot directly upgrade specific; use buy button
                            st.write("Upgrades are purchased via the 'Buy Upgrade' button, which offers random options.")
                            if st.button('Hide Details', key=f'hide_avail_{selected_glyph}_{root}_{branch}_{u}'):
                                st.session_state[f'detail_avail_{selected_glyph}_{root}_{branch}_{u}'] = False
