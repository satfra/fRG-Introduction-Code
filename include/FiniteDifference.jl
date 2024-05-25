using ShiftedArrays

function D_r(f, dir_coords, dir::Int)
  ddir = -(1:ndims(dir_coords) .== dir)
  df = (f .- ShiftedArray(f, ddir, default=NaN)) ./ (dir_coords .- ShiftedArray(dir_coords, ddir, default=NaN))
  selectdim(df, dir, size(dir_coords, dir)) .= selectdim(ShiftedArray(df, -ddir, default=NaN), dir, size(dir_coords, dir))
  return df
end

function D_r!(df, f, dir_coords, dir::Int)
  ddir = -(1:ndims(dir_coords) .== dir)
  df .= (f .- ShiftedArray(f, ddir, default=NaN)) ./ (dir_coords .- ShiftedArray(dir_coords, ddir, default=NaN))
  selectdim(df, dir, size(dir_coords, dir)) .= selectdim(ShiftedArray(df, -ddir, default=NaN), dir, size(dir_coords, dir))
  nothing
end

function D_l(f, dir_coords, dir::Int)
  ddir = (1:ndims(dir_coords) .== dir)
  df = (f .- ShiftedArray(f, ddir, default=NaN)) ./ (dir_coords .- ShiftedArray(dir_coords, ddir, default=NaN))
  selectdim(df, dir, 1) .= selectdim(ShiftedArray(df, -ddir, default=NaN), dir, 1)
  df
end

function D_l!(df, f, dir_coords, dir::Int)
  ddir = (1:ndims(dir_coords) .== dir)
  df .= (f .- ShiftedArray(f, ddir, default=0)) ./ (dir_coords .- ShiftedArray(dir_coords, ddir, default=0))
  selectdim(df, dir, 1) .= selectdim(ShiftedArray(df, -ddir, default=0), dir, 1)
  nothing
end

function D_lp!(df, f, dir_coords, dir::Int)
  ddir = (1:ndims(dir_coords) .== dir)
  df .+= (f .- ShiftedArray(f, ddir, default=0)) ./ (dir_coords .- ShiftedArray(dir_coords, ddir, default=0))
  selectdim(df, dir, 1) .+= selectdim(ShiftedArray(df, -ddir, default=0), dir, 1)
  nothing
end

function D_lm!(df, f, dir_coords, dir::Int)
  ddir = (1:ndims(dir_coords) .== dir)
  df .*= (f .- ShiftedArray(f, ddir, default=0)) ./ (dir_coords .- ShiftedArray(dir_coords, ddir, default=0))
  selectdim(df, dir, 1) .*= selectdim(ShiftedArray(df, -ddir, default=0), dir, 1)
  nothing
end

function sparsityNNnoDiag(p)
  grid = zeros(size(p.grid)[1:end-1]...)
  M = zeros(Float64, (size(grid)..., p.n_fields, size(grid)..., p.n_fields))
  for I in CartesianIndices(grid)
    # self-correlation
    M[I, :, I, :] .= 1.0
    for dir1 in 1:ndims(grid)
      u_v1 = CartesianIndex((1:ndims(grid) .== dir1)...)

      # correlation with all nearest neighbors
      if (I[dir1] != 1)
        M[I-u_v1, :, I, :] .= 1.0
        M[I, :, I-u_v1, :] .= 1.0
      end
      if (I[dir1] != size(grid, dir1))
        M[I+u_v1, :, I, :] .= 1.0
        M[I, :, I+u_v1, :] .= 1.0
      end
    end
  end
  return sparse(reshape(M, (p.n_fields * length(grid), p.n_fields * length(grid))))
end

function sparsityNN(p)
  grid = zeros(size(p.grid)[1:end-1]...)
  M = zeros(Float64, (size(grid)..., p.n_fields, size(grid)..., p.n_fields))
  for I in CartesianIndices(grid)
    # self-correlation
    M[I, :, I, :] .= 1.0
    for dir1 in 1:ndims(grid)
      u_v1 = CartesianIndex((1:ndims(grid) .== dir1)...)

      # correlation with all nearest neighbors
      if (I[dir1] != 1)
        M[I-u_v1, :, I, :] .= 1.0
        M[I, :, I-u_v1, :] .= 1.0
      end
      if (I[dir1] != size(grid, dir1))
        M[I+u_v1, :, I, :] .= 1.0
        M[I, :, I+u_v1, :] .= 1.0
      end
      # diagonal neighbors
      for dir2 in 1:ndims(grid)
        if dir2 == dir1
          continue
        end
        u_v2 = CartesianIndex((1:ndims(grid) .== dir2)...)
        if (all(Tuple(I - u_v1 - u_v2) .> 0) && all(Tuple(I - u_v1 - u_v2) .<= size(grid)))
          M[I-u_v1-u_v2, :, I, :] .= 1.0
          M[I, :, I-u_v1-u_v2, :] .= 1.0
        end
        if (all(Tuple(I + u_v1 - u_v2) .> 0) && all(Tuple(I + u_v1 - u_v2) .<= size(grid)))
          M[I+u_v1-u_v2, :, I, :] .= 1.0
          M[I, :, I+u_v1-u_v2, :] .= 1.0
        end
        if (all(Tuple(I - u_v1 + u_v2) .> 0) && all(Tuple(I - u_v1 + u_v2) .<= size(grid)))
          M[I-u_v1+u_v2, :, I, :] .= 1.0
          M[I, :, I-u_v1+u_v2, :] .= 1.0
        end
        if (all(Tuple(I + u_v1 + u_v2) .> 0) && all(Tuple(I + u_v1 + u_v2) .<= size(grid)))
          M[I+u_v1+u_v2, :, I, :] .= 1.0
          M[I, :, I+u_v1+u_v2, :] .= 1.0
        end
      end
    end
  end
  return sparse(reshape(M, (p.n_fields * length(grid), p.n_fields * length(grid))))
end

function sparsityNNN(p)
  grid = zeros(size(p.grid)[1:end-1]...)
  M = zeros(Float64, (size(grid)..., p.n_fields, size(grid)..., p.n_fields))
  for I in CartesianIndices(grid)
    # self-correlation
    M[I, :, I, :] .= 1.0
    for dir1 in 1:ndims(grid)
      u_v1 = CartesianIndex((1:ndims(grid) .== dir1)...)

      # correlation with all nearest neighbors
      if (I[dir1] != 1)
        M[I-u_v1, :, I, :] .= 1.0
        M[I, :, I-u_v1, :] .= 1.0
      end
      if (I[dir1] != size(grid, dir1))
        M[I+u_v1, :, I, :] .= 1.0
        M[I, :, I+u_v1, :] .= 1.0
      end

      # correlation with all next-nearest neighbors
      for dir2 in 1:ndims(grid)
        u_v2 = CartesianIndex((1:ndims(grid) .== dir2)...)
        if (all(Tuple(I - u_v1 - u_v2) .> 0) && all(Tuple(I - u_v1 - u_v2) .<= size(grid)))
          M[I-u_v1-u_v2, :, I, :] .= 1.0
          M[I, :, I-u_v1-u_v2, :] .= 1.0
        end
        if (all(Tuple(I + u_v1 - u_v2) .> 0) && all(Tuple(I + u_v1 - u_v2) .<= size(grid)))
          M[I+u_v1-u_v2, :, I, :] .= 1.0
          M[I, :, I+u_v1-u_v2, :] .= 1.0
        end
        if (all(Tuple(I - u_v1 + u_v2) .> 0) && all(Tuple(I - u_v1 + u_v2) .<= size(grid)))
          M[I-u_v1+u_v2, :, I, :] .= 1.0
          M[I, :, I-u_v1+u_v2, :] .= 1.0
        end
        if (all(Tuple(I + u_v1 + u_v2) .> 0) && all(Tuple(I + u_v1 + u_v2) .<= size(grid)))
          M[I+u_v1+u_v2, :, I, :] .= 1.0
          M[I, :, I+u_v1+u_v2, :] .= 1.0
        end
      end
    end
  end
  return sparse(reshape(M, (p.n_fields * length(grid), p.n_fields * length(grid))))
end

function sparsity_complex(p)
  grid = zeros(size(p.grid)[1:end-1]...)
  M = zeros(Float64, (2, size(grid)..., p.n_fields, 2, size(grid)..., p.n_fields))
  for I in CartesianIndices(grid)
    M[:, I, :, :, I, :] .= 1.0
    for dir in 1:ndims(grid)
      u_v = CartesianIndex((1:ndims(grid) .== dir)...)
      if (I[dir] != 1)
        M[:, I-u_v, :, :, I, :] .= 1.0
        M[:, I, :, :, I-u_v, :] .= 1.0
      end
      if (I[dir] != size(grid, dir))
        M[:, I+u_v, :, :, I, :] .= 1.0
        M[:, I, :, :, I+u_v, :] .= 1.0
      end
      if (I[dir] != 1 && I[dir] != size(grid, dir))
        M[:, I-u_v, :, :, I+u_v, :] .= 1.0
        M[:, I+u_v, :, :, I-u_v, :] .= 1.0
      end
    end
  end
  return sparse(reshape(M, (2 * p.n_fields * length(grid), 2 * p.n_fields * length(grid))))
end