function cothS(ϵ, T)
  return coth(ϵ / (2T))
end
function dcothS(ϵ, T)
  return -0.5 * csch(ϵ / (2 * T))^2 / T
end
function tanhS(ϵ, T)
  return tanh(ϵ / (2T))
end
function dtanhS(ϵ, T)
  return sech(ϵ / (2 * T))^2 / (2 * T)
end

function B1(m2_b, k, T)
  eb = sqrt(k^2 + m2_b)
  return (k * cothS(eb, T)) / (2.0 * eb)
end
function B2(m2_b, k, T)
  eb = sqrt(k^2 + m2_b)
  return (k^3 * cothS(eb, T)) / (4 * eb^3) - (k^3 * dcothS(eb, T)) / (4 * eb^2)
end

function FB12(m2_f, m2_b, k, T, mu)
  eb = sqrt(k^2 + m2_b)
  ef = sqrt(k^2 + m2_f)
  return (k^5 * (-3 * eb^14 + eb^12 * (19 * ef^2 - 9 * π^2 * T^2 + 9 * mu^2) + eb^4 * ((ef^2 + π^2 * T^2)^3 * (33 * ef^4 - 42 * ef^2 * π^2 * T^2 + 37 * π^4 * T^4) - (ef^2 + π^2 * T^2) * (57 * ef^6 + 67 * ef^4 * π^2 * T^2 + 139 * ef^2 * π^4 * T^4 - 63 * π^6 * T^6) * mu^2 + 2 * (5 * ef^6 + 103 * ef^4 * π^2 * T^2 + 303 * ef^2 * π^4 * T^4 + 13 * π^6 * T^6) * mu^4 - 2 * (9 * ef^4 + 38 * ef^2 * π^2 * T^2 + 13 * π^4 * T^4) * mu^6 + 3 * (23 * ef^2 - 21 * π^2 * T^2) * mu^8 - 37 * mu^10) - eb^2 * (π^2 * T^2 + (ef - mu)^2) * ((9 * ef^2 - 11 * π^2 * T^2) * (ef^2 + π^2 * T^2)^3 - 4 * (ef^2 + π^2 * T^2) * (4 * ef^4 + 29 * ef^2 * π^2 * T^2 - 3 * π^4 * T^4) * mu^2 + 2 * (-3 * ef^4 + 52 * ef^2 * π^2 * T^2 + 23 * π^4 * T^4) * mu^4 + 12 * (2 * ef^2 + π^2 * T^2) * mu^6 - 11 * mu^8) * (π^2 * T^2 + (ef + mu)^2) + eb^10 * (-51 * ef^4 + π^4 * T^4 + 90 * π^2 * T^2 * mu^2 + mu^4 + 30 * ef^2 * (π * T - mu) * (π * T + mu)) + (ef^2 + π^2 * T^2 - mu^2) * (ef^4 + 2 * ef^2 * (π * T - mu) * (π * T + mu) + (π^2 * T^2 + mu^2)^2)^3 + eb^8 * (75 * ef^6 + 23 * ef^4 * (-(π^2 * T^2) + mu^2) + ef^2 * (π^4 * T^4 - 38 * π^2 * T^2 * mu^2 + mu^4) + (π * T - mu) * (π * T + mu) * (35 * π^4 * T^4 + 246 * π^2 * T^2 * mu^2 + 35 * mu^4)) + eb^6 * (-65 * ef^8 + 55 * π^8 * T^8 + 188 * π^6 * T^6 * mu^2 - 118 * π^4 * T^4 * mu^4 + 188 * π^2 * T^2 * mu^6 + 55 * mu^8 + 28 * ef^6 * (-(π^2 * T^2) + mu^2) + 2 * ef^4 * (π^4 * T^4 - 38 * π^2 * T^2 * mu^2 + mu^4) + 4 * ef^2 * (5 * π^6 * T^6 - 43 * π^4 * T^4 * mu^2 + 43 * π^2 * T^2 * mu^4 - 5 * mu^6))) * cothS(eb, T)) / (8 * eb^3 * (π^2 * T^2 + (eb + ef - mu)^2)^2 * (π^2 * T^2 + (eb - ef + mu)^2)^2 * (π^2 * T^2 + (-eb + ef + mu)^2)^2 * (π^2 * T^2 + (eb + ef + mu)^2)^2) + (k^5 * (eb^6 - eb^4 * (3 * ef^2 - π^2 * T^2 + mu^2) + eb^2 * (3 * ef^4 + 2 * ef^2 * π^2 * T^2 - π^4 * T^4 - 2 * (ef^2 + 5 * π^2 * T^2) * mu^2 - mu^4) - (π^2 * T^2 + (ef - mu)^2) * (ef^2 + π^2 * T^2 - mu^2) * (π^2 * T^2 + (ef + mu)^2)) * dcothS(eb, T)) / (8 * eb^2 * (π^2 * T^2 + (eb + ef - mu)^2) * (π^2 * T^2 + (eb - ef + mu)^2) * (π^2 * T^2 + (-eb + ef + mu)^2) * (π^2 * T^2 + (eb + ef + mu)^2)) + (k^5 * (eb^2 - ef^2 - 2 * ef * π * T + π^2 * T^2 + 2 * (ef + π * T) * mu - mu^2) * (eb^2 - ef^2 + π^2 * T^2 - 2 * π * T * mu - mu^2 + 2 * ef * (π * T + mu)) * tanhS(ef - mu, T)) / (8 * ef * (π^2 * T^2 + (eb + ef - mu)^2)^2 * (π^2 * T^2 + (eb - ef + mu)^2)^2) + (k^5 * (-eb^2 + ef^2 - π^2 * T^2 - 2 * π * T * mu + mu^2 + 2 * ef * (-(π * T) + mu)) * (-eb^2 + ef^2 - π^2 * T^2 + 2 * π * T * mu + mu^2 + 2 * ef * (π * T + mu)) * tanhS(ef + mu, T)) / (8 * ef * (π^2 * T^2 + (-eb + ef + mu)^2)^2 * (π^2 * T^2 + (eb + ef + mu)^2)^2)
end
function FB21(m2_f, m2_b, k, T, mu)
  eb = sqrt(k^2 + m2_b)
  ef = sqrt(k^2 + m2_f)
  return (k^5 * (eb^12 + 2 * eb^10 * (-3 * ef^2 - π^2 * T^2 + mu^2) + eb^4 * ((ef^2 + π^2 * T^2)^2 * (15 * ef^4 + 14 * ef^2 * π^2 * T^2 - 17 * π^4 * T^4) + 4 * (-11 * ef^6 - 7 * ef^4 * π^2 * T^2 + 11 * ef^2 * π^4 * T^4 + 23 * π^6 * T^6) * mu^2 + 2 * (13 * ef^4 - 22 * ef^2 * π^2 * T^2 + 173 * π^4 * T^4) * mu^4 + 4 * (5 * ef^2 + 23 * π^2 * T^2) * mu^6 - 17 * mu^8) + 2 * eb^2 * (-((ef^2 + π^2 * T^2)^4 * (3 * ef^2 + π^2 * T^2)) + (ef^2 + π^2 * T^2)^2 * (13 * ef^4 - 22 * ef^2 * π^2 * T^2 + 45 * π^4 * T^4) * mu^2 + 2 * (-11 * ef^6 - 7 * ef^4 * π^2 * T^2 + 11 * ef^2 * π^4 * T^4 + 23 * π^6 * T^6) * mu^4 + 2 * (9 * ef^4 + 34 * ef^2 * π^2 * T^2 - 23 * π^4 * T^4) * mu^6 - (7 * ef^2 + 45 * π^2 * T^2) * mu^8 + mu^10) + eb^8 * (15 * ef^4 - 17 * π^4 * T^4 - 90 * π^2 * T^2 * mu^2 - 17 * mu^4 + 14 * ef^2 * (π * T - mu) * (π * T + mu)) + (ef^2 + π^2 * T^2 - 2 * π * T * mu - mu^2) * (ef^2 + π^2 * T^2 + 2 * π * T * mu - mu^2) * (ef^4 + 2 * ef^2 * (π * T - mu) * (π * T + mu) + (π^2 * T^2 + mu^2)^2)^2 - 4 * eb^6 * (5 * ef^6 + 9 * ef^4 * (π * T - mu) * (π * T + mu) - ef^2 * (5 * π^4 * T^4 + 34 * π^2 * T^2 * mu^2 + 5 * mu^4) + (π * T - mu) * (π * T + mu) * (7 * π^4 * T^4 + 30 * π^2 * T^2 * mu^2 + 7 * mu^4))) * cothS(eb, T)) / (4 * eb * (π^2 * T^2 + (eb + ef - mu)^2)^2 * (π^2 * T^2 + (eb - ef + mu)^2)^2 * (π^2 * T^2 + (-eb + ef + mu)^2)^2 * (π^2 * T^2 + (eb + ef + mu)^2)^2) + (k^5 * (-eb^2 - π^2 * T^2 + (ef - mu)^2) * dtanhS(ef - mu, T)) / (16 * ef^2 * (π^2 * T^2 + (eb + ef - mu)^2) * (π^2 * T^2 + (eb - ef + mu)^2)) + (k^5 * (-eb^2 + (ef - π * T + mu) * (ef + π * T + mu)) * dtanhS(ef + mu, T)) / (16 * ef^2 * (π^2 * T^2 + (-eb + ef + mu)^2) * (π^2 * T^2 + (eb + ef + mu)^2)) + (k^5 * (eb^6 + eb^4 * (-5 * ef^2 + 3 * π^2 * T^2 + 8 * ef * mu - 3 * mu^2) - (π^2 * T^2 + (ef - mu)^2) * (3 * ef^4 - π^4 * T^4 - 10 * ef^3 * mu + mu^4 + 6 * ef * (π * T - mu) * mu * (π * T + mu) - 6 * ef^2 * (π^2 * T^2 - 2 * mu^2)) + eb^2 * (7 * ef^4 + 3 * π^4 * T^4 - 24 * ef^3 * mu - 2 * π^2 * T^2 * mu^2 - 16 * ef * mu^3 + 3 * mu^4 + 2 * ef^2 * (π^2 * T^2 + 15 * mu^2))) * tanhS(ef - mu, T)) / (16 * ef^3 * (π^2 * T^2 + (eb + ef - mu)^2)^2 * (π^2 * T^2 + (eb - ef + mu)^2)^2) + (k^5 * (eb^6 - eb^4 * (-3 * π^2 * T^2 + (ef + mu) * (5 * ef + 3 * mu)) - (π^2 * T^2 + (ef + mu)^2) * (3 * ef^4 - π^4 * T^4 + 10 * ef^3 * mu + mu^4 - 6 * ef^2 * (π^2 * T^2 - 2 * mu^2) + 6 * ef * mu * (-(π^2 * T^2) + mu^2)) + eb^2 * (7 * ef^4 + 3 * π^4 * T^4 + 24 * ef^3 * mu - 2 * π^2 * T^2 * mu^2 + 16 * ef * mu^3 + 3 * mu^4 + 2 * ef^2 * (π^2 * T^2 + 15 * mu^2))) * tanhS(ef + mu, T)) / (16 * ef^3 * (π^2 * T^2 + (-eb + ef + mu)^2)^2 * (π^2 * T^2 + (eb + ef + mu)^2)^2)
end

function lB0(m2_b, k, T, d)
  pref = 2 / (d - 1)
  return pref * B1(m2_b, k, T)
end
function lB1(m2_b, k, T, d)
  pref = 2 / (d - 1)
  return pref * B2(m2_b, k, T)
end
function L_11(m2_f, m2_b, k, T, mu, d)
  pref = 2.0 / (d - 1.0)
  return pref * (FB21(m2_f, m2_b, k, T, mu) + FB12(m2_f, m2_b, k, T, mu))
end