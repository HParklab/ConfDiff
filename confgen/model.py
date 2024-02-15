import torch

from torch import nn
from utils import *
from rdkit_type import ATOM_TYPE, BOND_TYPE


class E_GCL(nn.Module):

    def __init__(self, input_nf, output_nf, hidden_nf,
                 edges_in_d=0, act_fn=nn.SiLU(), 
                 residual=True, normalize=True,
                 tanh=True, norm_constant=1,
                 attention=True, coords_agg='sum'
                ):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2 + edges_in_d
        self.residual = residual
        self.normalize = normalize
        self.norm_constant = norm_constant
        
        self.attention = attention
        self.coords_agg = coords_agg
        self.tanh = tanh

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1)
                )
            
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        if self.tanh:
            raise NotImplementedError
            coord_mlp.append(nn.Tanh())

        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer)
    

    def edge_model(self, h, edge_attr):
        B, N, _ = h.shape
        source = h[:, None].tile([1, N, 1, 1])
        target = h[:, :, None].tile([1, 1, N, 1])
        edge_attr = edge_attr.view(B, N, N, -1)

        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=3)
        else:
            out = torch.cat([source, target, edge_attr], dim=3)
        out = out.reshape([B*N*N, -1])
        out = self.edge_mlp(out)
        
        if self.attention:
            att_val = self.att_mlp(out)
            att_val = nn.functional.softmax(att_val, dim=1)    
            out = out * att_val
        out = out.reshape([B, N, N, -1])
        return out

    def node_model(self, h, edge_feat, mask):
        assert len(h.shape) == 3
        assert len(edge_feat.shape) == 4
        B, N, _ = h.shape
        
        # aggregation method
        agg = torch.sum(edge_feat, axis=2)
        agg = torch.cat([h, agg], dim=-1)
        agg = agg * mask[:, :, None]
        agg = agg.reshape([B*N, -1])
        
        out = self.node_mlp(agg)
        out = out.reshape([B, N, -1])

        if self.residual:
            out = h + out
        return out

    def coord_model(self, h, coord, coord_diff, mask, edge_attr):
        assert len(coord.shape) == 3
        assert len(coord_diff.shape) == 4
        assert len(mask.shape) == 2
        assert len(edge_attr.shape) == 4
        B, N, D = coord.shape
        source = h[:, None].tile([1, N, 1, 1])
        target = h[:, :, None].tile([1, 1, N, 1])
        
        mask_2d = mask[:, :, None] * mask[:, None, :]
        input_tensor = torch.cat([source, target, edge_attr], dim=3)
        input_tensor = input_tensor.reshape([B, N**2, -1])

        coord_diff = coord_diff.reshape([B, N**2, D])
        trans = coord_diff * self.coord_mlp(input_tensor)
        trans = mask_2d[..., None]*trans.reshape([B, N, N, D])

        if self.coords_agg=='sum':
            agg = torch.sum(trans, axis=2)
        elif self.coords_agg=='mean':
            agg = torch.sum(trans, axis=2) / (torch.sum(
                mask_2d, axis=2, keepdim=True) + 1e-10)

        coord = coord + agg
        return coord


    def forward(self, h, coord, mask=None, edge_attr=None, device='cpu'):
        if mask is None:
            mask = torch.ones(h.shape[:2]).to(device)
        coord *= mask[..., None]
        h *= mask[..., None]

        mask_2d = mask[:, :, None] * mask[:, None, :]
        radial, coord_diff = coord2radial(coord, norm_constant=self.norm_constant)
        radial *= mask_2d[..., None]
        coord_diff *= mask_2d[..., None]
        edge_attr *= mask_2d[..., None]

        edge_feat = self.edge_model(h, edge_attr)
        edge_feat *= mask_2d[..., None]
        
        h = self.node_model(h, edge_feat, mask)
        h *= mask[:, :, None]

        coord = self.coord_model(h, coord, coord_diff, mask, edge_attr)
        coord *= mask[:, :, None]
        
        return h, coord, edge_attr


class EGNN(nn.Module):
    
    def __init__(
                self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0,
                device='cpu', act_fn=nn.SiLU(), n_layers=4,
                residual=True, normalize=False, tanh=False, norm_constant=1,
                attention=True, coords_agg='sum'
                ):

        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.norm_constant = norm_constant

        self.node_encoding = SinusoidsEmbeddingNew()
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, norm_constant=norm_constant,
                                                normalize=normalize, tanh=tanh, attention=attention, coords_agg=coords_agg))
        self.to(self.device)

    def forward(self, h, x, edge_attr, mask=None):
        B, N, _ = h.shape
        h = h.reshape([B*N, -1])
        h = self.embedding_in(h)
        h = h.reshape([B, N, -1])
        
        distance, _ = coord2radial(x, self.norm_constant)
        distance = self.node_encoding(distance)#.to(self.device)
        distance = distance.view(B, N, N, -1)
        edge_attr = torch.cat([edge_attr, distance], dim=-1)

        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, x, mask=mask,
                    edge_attr=edge_attr, device=self.device)
             
        h = h.reshape([B*N, -1])
        h = self.embedding_out(h)
        h = h.reshape([B, N, -1])
        return h, x
    


class AtomEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(len(atom_type_list), embed_dim)
                for atom_type_list in ATOM_TYPE.values()
            ]
        )
        self.num_features = len(self.embeddings)
        self.scale = 1.0 / math.sqrt(self.num_features)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        feats_embed = 0.0
        for i in range(self.num_features):
            feats_embed += self.scale * self.embeddings[i](feats[..., i])
        return feats_embed 


class BondEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(len(bond_type_list), embed_dim)
                for bond_type_list in BOND_TYPE.values()
            ]
        )
        self.num_features = len(self.embeddings)
        self.scale = 1.0 / math.sqrt(self.num_features)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        feats_embed = 0.0
        for i in range(self.num_features):
            feats_embed += self.scale * self.embeddings[i](feats[..., i])
        return feats_embed 



class Dynamics(nn.Module):

    def __init__(self, device, args, 
                 hidden_nf=256, embedding_nf=128, norm_constant=1,
                 normalize=True, tanh=False, 
                 ):

        super().__init__()
        self.device = device
        self.args = args
        
        if args.noise_schedule=='linear':
            betas = torch.linspace(start=args.beta_1, end=args.beta_T, steps=args.T)
            self.alpha_bars = torch.cumprod(1 - betas, dim = 0).to(device)
        elif args.noise_schedule=='cosine':
            betas = get_cosine_betas(args.T)
            self.alpha_bars = torch.cumprod(1 - betas, dim = 0).to(device)
            
        self.bond_embedding = BondEmbedding(embedding_nf)
        self.atom_embedding = AtomEmbedding(embedding_nf)
    
        self.in_node_nf = embedding_nf + 1
        self.hidden_nf = hidden_nf
        self.embedding_nf = embedding_nf
        self.device = device
        
        try:
            self.scale_eps = args.scale_eps
        except:
            self.scale_eps = False 
                
        self.layers = []
        for i in range(args.num_egnn_layers):
            layer = []
            egnn = EGNN(
                in_node_nf  = self.in_node_nf,
                hidden_nf   = self.hidden_nf, 
                out_node_nf = self.in_node_nf,
                in_edge_nf  = self.embedding_nf + 12,
                n_layers    = args.num_gcl_layers,
                normalize   = normalize,
                tanh        = tanh,
                attention   = args.attention,
                coords_agg  = args.coords_agg,
                norm_constant = norm_constant
            ).to(device)
            layer.append(egnn)
            layer.append(nn.LayerNorm(self.in_node_nf))
            self.layers.append(layer)

        self.layers_pytorch = nn.ModuleList([l for sublist in self.layers for l in sublist])
        self.to(device = self.device)

    def loss_fn(self, data, idx=None):
        output, epsilon = self.forward(data, idx=idx, get_target=True)
        atom_mask = data['atom_mask'].type(torch.float32) 

        errors = output - epsilon
        errors *= atom_mask[..., None]
        losses = torch.mean(errors**2, dim=(-1))
        loss = torch.sum(losses) / (torch.sum(atom_mask) + 1e-10)
        return loss

    def forward(self, input_feats, idx=None, get_target=False):
        
        atom_mask = input_feats['atom_mask'].type(torch.float32) 
        edge_mask = atom_mask[:, None, :] * atom_mask[:, :, None]
        
        if idx == None:
            
            idx = torch.randint(0, len(self.alpha_bars), (input_feats['positions'].size(0), )).to(device = self.device)
            used_alpha_bars = self.alpha_bars[idx][:, None, None]
            
            epsilon = torch.randn_like(input_feats['positions']) * atom_mask[..., None]
            epsilon = remove_mean_with_mask(epsilon, atom_mask[..., None])
            
            x_tilde = torch.sqrt(used_alpha_bars) * input_feats['positions'] + torch.sqrt(1 - used_alpha_bars) * epsilon

        else:
            
            idx = torch.Tensor([idx for _ in range(input_feats['positions'].size(0))]).to(device = self.device).long()
            x_tilde = input_feats['positions']
        
        
        atom_position = x_tilde.type(torch.float32)
        # assert_mean_zero_with_mask(atom_position, atom_mask[..., None])

        curr_pos = atom_position.clone() # [B, N, D]
        t = idx
        B, N, _ = atom_position.shape
        
        edge_attr = input_feats['bond_feats']
        edge_attr = edge_attr.view(B, N*N, -1)
        edge_attr = self.bond_embedding(edge_attr)
        edge_attr = edge_attr.view(B, N, N, -1)
        edge_attr *= edge_mask[..., None]
        
        node_attr = input_feats['atom_feats']
        node_attr = self.atom_embedding(node_attr)
        node_time = torch.tile((t/len(self.alpha_bars)).view(-1,1,1), [1, N, 1])
        node_attr = torch.cat([node_attr, node_time], dim=2)

        for layer in self.layers:
            node_attr *= atom_mask[..., None]
            curr_pos *= atom_mask[..., None]
            
            egnn, norm = layer
            node_attr, curr_pos = egnn(node_attr, curr_pos, edge_attr, mask=atom_mask)
            node_attr *= atom_mask[..., None]
            node_attr = norm(node_attr)
        
        if self.scale_eps:
            cum_a_t = self.alpha_bars[t[:, None, None]]
            eps_theta_val = atom_position - curr_pos * cum_a_t
            eps_theta_val = eps_theta_val / torch.sqrt(1 - cum_a_t)
            eps_theta_val = eps_theta_val * atom_mask[..., None]
        
        else:
            eps_theta_val = curr_pos - atom_position
            eps_theta_val = eps_theta_val.reshape(atom_position.shape)
            eps_theta_val = eps_theta_val * atom_mask[..., None]
            
        # eps_theta_val = remove_mean_with_mask(eps_theta_val, atom_mask[..., None])

        return (eps_theta_val, epsilon) if get_target else eps_theta_val