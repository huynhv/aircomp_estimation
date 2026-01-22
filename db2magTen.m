% DB2MAGTEN Converts a quantity from dB to decimal.
function out = db2magTen(db_val)
    out = db2mag(db_val).^2;
end